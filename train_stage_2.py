import torch
import datasets, models, losses, utils
from tqdm import tqdm
from models.FeedbackLoopModule import FeedbackLoopModule
import torch.nn.functional as F


def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser(description='CrimSniffer: Train Stage 2')
    parser.add_argument('--dataset', type=str, required=True, help='Path to training dataset.')
    parser.add_argument('--dataset_validation', type=str, default=None, help='Path to validation dataset.')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--resume', type=str, default=None, help='Path to load model weights.')
    parser.add_argument('--resume_CE', type=str, default=None, help='Path to load Component Embedding model weights. Required if --resume is not given. Skipped if --resume is given.')
    parser.add_argument('--output', type=str, default=None, help='Path to save weights.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--comet', type=str, default=None, help='comet.ml API')
    parser.add_argument('--comet_log_image', type=str, default=None, help='Path to model input image to be inference and log the result to comet.ml. Skipped if --comet is not given.')
    parser.add_argument('--self_supervised_loss_weight', type=float, default=1.0, help='Weight for the self-supervised loss')
    parser.add_argument('--refined_image_loss_weight', type=float, default=1.0, help='Weight for the refined image loss')
    args = parser.parse_args()
    return args

def validation_parser(args):
    if args.resume:
        if args.resume_CE: print('args.resume_CE will be skipped.')
    else:
        assert args.resume_CE, "Both args.resume and args.resume_CE can't be None."
    if not args.comet:
        if args.comet_log_image: print('args.comet_log_image will be skipped.')

def main(args):

    device = torch.device(args.device)
    print(f'Device : {device}')
    self_supervised_loss_weight = args.self_supervised_loss_weight
    refined_image_loss_weight = args.refined_image_loss_weight
    self_supervised_module = losses.SelfSupervisedModule()
    feed_back_module = FeedbackLoopModule()    

    if args.comet:
        from comet_ml import Experiment
        experiment = Experiment(
            api_key=args.comet,
            project_name="crimsniffer-training-stage-2",
            workspace="ahmad786writes",
            log_code=True
        )

        if args.comet_log_image:
            log_image_sketch = datasets.dataloader.load_one_sketch(args.comet_log_image).unsqueeze(0).to(device)

    model = models.CrimSniffer(
        CE=True, CE_encoder=True, CE_decoder=False,
        FM=True, FM_decoder=True,
        IS=True, IS_generator=True, IS_discriminator=True,
        manifold=False
    )

    if args.comet:
        experiment.set_model_graph(model)

    if args.resume:
        model.load(args.resume, map_location=device)
    else:
        model.CE.load(args.resume_CE, map_location=device)

    model.to(device)

    train_dataloader = datasets.dataloader.dataloader(args.dataset, batch_size=args.batch_size, load_photo=True, augmentation=False)

    if args.dataset_validation:
        validation_dataloader = datasets.dataloader.dataloader(args.dataset_validation, batch_size=args.batch_size, load_photo=True)

    for key, component in model.CE.components.items():
        for param in component.parameters():
            param.requires_grad = False

    optimizer_generator = torch.optim.Adam( list(model.FM.parameters()) + list(model.IS.G.parameters()) , lr=0.0002, betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam( list(model.IS.D1.parameters()) + list(model.IS.D2.parameters()) + list(model.IS.D3.parameters()) , lr=0.0002, betas=(0.5, 0.999))

    l1 = losses.L1()
    bce = losses.BCE()
    perceptual = losses.Perceptual(device=args.device)
    style_loss = losses.StyleLoss(device=args.device)
    combined_loss = losses.CombinedLoss(alpha=0.5)

    label_real = model.IS.label_real
    label_fake = model.IS.label_fake

    for epoch in range(args.epochs):

        running_loss = {
            'loss_G' : 0,
            'loss_D' : 0
        }

        model.train()
        for sketches, photos in tqdm(train_dataloader, desc=f'Epoch - {epoch+1} / {args.epochs}'):

            sketches = sketches.to(device)
            photos = photos.to(device)

            latents = model.CE.encode(model.CE.crop(sketches))
            spatial_map = model.FM.merge(model.FM.decode(latents))
            fake_photos = model.IS.generate(spatial_map)
            perturbated_images = feed_back_module(fake_photos)
        
            optimizer_generator.zero_grad()
            # total_loss = 0.0
            # for gen_img in perturbated_images:
            #     loss = combined_loss(gen_img, fake_photos)
            #     total_loss += loss.item()
            # perturbated_images_loss = total_loss / len(perturbated_images)
            loss_G_L1 = l1.compute(fake_photos, photos)
            loss_perceptual = perceptual.compute(fake_photos, photos)
            patches = model.IS.discriminate(spatial_map, fake_photos)
            loss_G_BCE = torch.tensor([bce.compute(patch, torch.full(patch.shape, label_real, dtype=torch.float, requires_grad=True).to(device)) for patch in patches], dtype=torch.float, requires_grad=True).sum()
            loss_gram_matrix_loss = style_loss(fake_photos, photos)
            # refined_image_loss = F.mse_loss(refined_image, fake_photos)
            # self_supervised_loss = self_supervised_module(sketches, fake_photos)
            loss_G = loss_perceptual + 10 * loss_G_L1 + loss_G_BCE + loss_gram_matrix_loss # + refined_image_loss_weight * perturbated_images_loss + self_supervised_loss_weight * self_supervised_loss
            # loss_G = loss_perceptual + 10 * loss_G_L1 + loss_G_BCE
            loss_G.backward()
            optimizer_generator.step()

            optimizer_discriminator.zero_grad()
            patches = model.IS.discriminate(spatial_map.detach(), fake_photos.detach())
            loss_D_fake = torch.tensor([bce.compute(patch, torch.full(patch.shape, label_fake, dtype=torch.float, requires_grad=True).to(device)) for patch in patches], dtype=torch.float, requires_grad=True).sum()
            patches = model.IS.discriminate(spatial_map.detach(), photos.detach())
            loss_D_real = torch.tensor([bce.compute(patch, torch.full(patch.shape, label_real, dtype=torch.float, requires_grad=True).to(device)) for patch in patches], dtype=torch.float, requires_grad=True).sum()
            loss_D = loss_D_fake + loss_D_real
            loss_D.backward()
            optimizer_discriminator.step()

            iteration_loss = {
                'loss_G_it' : loss_G.item(),
                'loss_D_it' : loss_D.item()
            }

            for key, loss in iteration_loss.items():
                running_loss[key[:-3]] = loss * len(sketches) / len(train_dataloader.dataset)

            if args.comet:
                experiment.log_metrics(iteration_loss)

        if args.dataset_validation:
            validation_running_loss = {
                'val_loss_G' : 0,
                'val_loss_D' : 0
            }

            model.eval()
            with torch.no_grad():
                for sketches, photos in tqdm(validation_dataloader, desc=f'Validation Epoch - {epoch+1} / {args.epochs}'):

                    sketches = sketches.to(device)
                    photos = photos.to(device)

                    latents = model.CE.encode(model.CE.crop(sketches))
                    spatial_map = model.FM.merge(model.FM.decode(latents))
                    fake_photos = model.IS.generate(spatial_map)
                    perturbated_images = feed_back_module(fake_photos)
                    # total_loss = 0.0
                    # for gen_img in perturbated_images:
                    #     loss = combined_loss(gen_img, fake_photos)
                    #     total_loss += loss.item()
                    # perturbated_images_loss = total_loss / len(perturbated_images)
                    loss_G_L1 = l1.compute(fake_photos, photos)
                    loss_perceptual = perceptual.compute(fake_photos, photos)
                    patches = model.IS.discriminate(spatial_map, fake_photos)
                    loss_gram_matrix_loss = style_loss(fake_photos, photos)
                    # refined_image_loss = F.mse_loss(refined_image, fake_photos)
                    # self_supervised_loss = self_supervised_module(sketches, fake_photos)
                    loss_G = loss_perceptual + 10 * loss_G_L1 + loss_G_BCE + loss_gram_matrix_loss # + refined_image_loss_weight * perturbated_images_loss + self_supervised_loss_weight * self_supervised_loss 
                    # loss_G = loss_perceptual + 10 * loss_G_L1 + loss_G_BCE

                    patches = model.IS.discriminate(spatial_map.detach(), fake_photos.detach())
                    loss_D_fake = torch.tensor([bce.compute(patch, torch.full(patch.shape, label_fake, dtype=torch.float).to(device)) for patch in patches], dtype=torch.float).sum()
                    patches = model.IS.discriminate(spatial_map.detach(), photos.detach())
                    loss_D_real = torch.tensor([bce.compute(patch, torch.full(patch.shape, label_real, dtype=torch.float).to(device)) for patch in patches], dtype=torch.float).sum()
                    loss_D = loss_D_fake + loss_D_real

                    validation_iteration_loss = {
                        'val_loss_G_it' : loss_G.item(),
                        'val_loss_D_it' : loss_D.item()
                    }

                    for key, loss in validation_iteration_loss.items():
                        validation_running_loss[key[:-3]] = loss * len(sketches) / len(validation_dataloader.dataset)

                    if args.comet:
                        experiment.log_metrics(validation_iteration_loss)

        def print_dict_loss(dict_loss):
            for key, loss in dict_loss.items():
                print(f'Loss {key:12} : {loss:.6f}')

        print()
        print(f'Epoch - {epoch+1} / {args.epochs}')
        print_dict_loss(running_loss)
        if args.dataset_validation: print_dict_loss(validation_running_loss)
        print()

        if args.comet:
            experiment.log_metrics(running_loss, step=epoch+1)
            if args.dataset_validation: 
                experiment.log_metrics(validation_running_loss, step=epoch+1)
            if args.comet_log_image:
                generated_image = model(log_image_sketch)
                generated_image_pil = utils.convert.tensor2PIL(generated_image[0])
                experiment.log_image(generated_image_pil, name="Generated Image", step=epoch+1)


        if args.output:
            model.save(args.output)

    if args.comet:
        experiment.end()

if __name__ == '__main__':
    args = get_args_parser()
    print(args)
    validation_parser(args)
    main(args)
