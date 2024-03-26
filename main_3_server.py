import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from datasets import load_dataset
from accelerate import Accelerator
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
import timeit

import json
import argparse




device = "cuda" if torch.cuda.is_available() else "cpu"

# Set up argument parsing
parser = argparse.ArgumentParser(description='DDPM Training Script')
parser.add_argument('--dataname', type=str, required=True, help='A dataset name to process (e.g., NORMAL, PNEUMONIA, COVID)')
# parser.add_argument('--RANDOM_SEED', type=int, default=42, help='Random seed for initialization')
parser.add_argument('--IMG_SIZE', type=int, default=128, help='Image size (e.g., 128 for 128x128 images)')
parser.add_argument('--BATCH_SIZE', type=int, default=16, help='Batch size for training')
parser.add_argument('--LEARNING_RATE', type=float, default=1e-4, help='Learning rate for the optimizer')
parser.add_argument('--NUM_EPOCHS', type=int, default=215, help='Number of epochs for training')
parser.add_argument('--NUM_GENERATE_IMAGES', type=int, default=200, help='Number of images to generate')
parser.add_argument('--NUM_TIMESTEPS', type=int, default=4000, help='Number of timesteps for the diffusion process')

args = parser.parse_args()

# Use the arguments
dataname = args.dataname # dataname = "NORMAL", "PNEUMONIA", "COVID"
# Hyperparameters
# RANDOM_SEED = args.RANDOM_SEED
IMG_SIZE = args.IMG_SIZE
BATCH_SIZE = args.BATCH_SIZE
LEARNING_RATE = args.LEARNING_RATE
NUM_EPOCHS = args.NUM_EPOCHS
NUM_GENERATE_IMAGES = args.NUM_GENERATE_IMAGES
NUM_TIMESTEPS = args.NUM_TIMESTEPS

# RANDOM_SEED = args.RANDOM_SEED #42
RANDOM_SEED = 42
# IMG_SIZE = args.IMG_SIZE #128
# BATCH_SIZE = args.BATCH_SIZE #16
LEARNING_RATE = 1e-4 # 1e-4
# NUM_EPOCHS = args.NUM_EPOCHS #215
# NUM_GENERATE_IMAGES = args.NUM_GENERATE_IMAGES #200
# NUM_TIMESTEPS = args.NUM_TIMESTEPS #4000
MIXED_PRECISION = "fp16"
GRADIENT_ACCUMULATION_STEPS = 1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


dir_path = os.getcwd()
save_path = f"{dir_path}/save_model/Server_BATCHSIZE{BATCH_SIZE}_TIMESTEPS{NUM_TIMESTEPS}_EPOCHS{NUM_EPOCHS}_LEARNING_RATE{LEARNING_RATE}_checkpoint_{dataname}.pth"
dataset = load_dataset(f"{dir_path}/chest_X_ray_{dataname}", split="train", trust_remote_code=True)


# Data aumentation
preprocess = transforms.Compose(
[
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# U-Net
model = UNet2DModel(
    sample_size=IMG_SIZE,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D"
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D"
    )
)
model = model.to(device)

noise_scheduler = DDPMScheduler(num_train_timesteps=NUM_TIMESTEPS)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(train_dataloader)*NUM_EPOCHS
)

accelerator = Accelerator(
    mixed_precision=MIXED_PRECISION,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
)

model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)

############################################################
# Train model
start = timeit.default_timer()

training_metrics = {
    'losses': [],
    'mses': [],
    'learning_rates': [],
}

for epoch in tqdm(range(NUM_EPOCHS), position=0, leave=True):
    model.train()
    train_running_loss = 0
    mse_sum = 0  # Initialize sum of MSE for the epoch
    for idx, batch in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        clean_images = batch["images"].to(device)
        noise = torch.randn(clean_images.shape).to(device)
        last_batch_size = len(clean_images)

        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (last_batch_size,)).to(device)
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        with accelerator.accumulate(model):
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            accelerator.backward(loss)

            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        train_running_loss += loss.item()
        mse_sum += F.mse_loss(noise_pred, noise, reduction='sum').item()
    train_loss = train_running_loss / (idx+1)
    # mse_epoch = mse_sum / (idx + 1) * BATCH_SIZE  # Assuming each batch has BATCH_SIZE samples
    mse_epoch = mse_sum / (len(train_dataloader.dataset))  # Adjust this based on how you calculate MSE

    # Append the metrics for the current epoch to the dictionary
    training_metrics['losses'].append(train_loss)
    training_metrics['mses'].append(mse_epoch)
    training_metrics['learning_rates'].append(lr_scheduler.get_last_lr()[0])
    train_learning_rate = lr_scheduler.get_last_lr()[0]
    print("-"*30)
    print(f"Train Loss EPOCH: {train_running_loss}: {idx+1}")
    print(f"Train Loss EPOCH: {epoch+1}: {train_loss:.4f}")
    print(f"Train Learning Rate EPOCH: {epoch+1}: {train_learning_rate}")
    # if epoch%10 == 0:
    #     sample_image_generation(model, noise_scheduler, NUM_GENERATE_IMAGES, RANDOM_SEED, NUM_TIMESTEPS)
    print("-"*30)

stop = timeit.default_timer()
print(f"Training Time: {stop-start:.2f}s")


# Check if the directory exists, if not, create it
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

filename = f"{dir_path}/server_training_metrics_BATCHSIZE{BATCH_SIZE}_TIMESTEPS{NUM_TIMESTEPS}_EPOCHS{NUM_EPOCHS}_LEARNING_RATE{LEARNING_RATE}_{dataname}.json"

# Save the metrics to the file
with open(filename, 'w') as f:
    json.dump(training_metrics, f)

print(f"Metrics saved to {filename}")

############################################################
# Saving checkpoints
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": lr_scheduler.state_dict(),
}
torch.save(checkpoint, save_path)

############################################################
# Generate Images
def sample_image_generation(model, noise_scheduler, num_generate_images, random_seed, num_timesteps, save_dir, dataname):
    pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

    images = pipeline(
        batch_size=num_generate_images,
        # generator=torch.manual_seed(random_seed),
        num_inference_steps=num_timesteps
    ).images

    for i, image in enumerate(images):
        if not os.path.exists(save_dir):
            # If the directory does not exist, create it
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"generated_image_{dataname}_{i+1}.png")
        if not isinstance(image, Image.Image):
            # Convert tensor image to PIL Image only if necessary
            pil_image = transforms.ToPILImage()(image)
        else:
            # If it's already a PIL image, no need to convert
            pil_image = image
        pil_image.save(save_path)

    fig = plt.figure()
    # for i in range(1, num_generate_images+1):
    for i in range(1, 10):
        fig.add_subplot(3, 3, i)
        plt.imshow(images[i-1])
    plt.show()

NUM_GENERATE_IMAGES = 100
RANDOM_SEED = 42
save_dir = f"{dir_path}/save_images_BATCHSIZE{BATCH_SIZE}_TIMESTEPS{NUM_TIMESTEPS}_EPOCHS{NUM_EPOCHS}_{dataname}"
sample_image_generation(model, noise_scheduler, NUM_GENERATE_IMAGES, RANDOM_SEED, NUM_TIMESTEPS, save_dir, dataname)