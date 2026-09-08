import torch

print("PYTORCH DEVICE CHECK")
print("====================")

print()

print("PyTorch version:")
print(torch.__version__)

print()

print("CUDA available:")
print(torch.cuda.is_available())

if torch.cuda.is_available():

    print()

    print("GPU name:")
    print(torch.cuda.get_device_name(0))

    print()

    print("CUDA version:")
    print(torch.version.cuda)

else:

    print()

    print("No NVIDIA GPU detected.")
    print("Training will use CPU.")