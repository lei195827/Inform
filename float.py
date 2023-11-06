import torch
from thop import profile
from models import Informer

pre_len = 60

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path="weights/informer_epoch_8.pth", pre_len=60, enc_in=8):
    model = Informer(out_len=pre_len, enc_in=enc_in)
    model.load_state_dict(torch.load(model_path))  # Load the trained model weights
    model.eval()
    model.to(device)
    return model


# Load your model
model = load_model()
model.eval()

# Create random input tensors
batch_size = 1  # Example batch size, adjust as needed
s_len = 120  # Input sequence length
feature_dim = 11   # Input feature dimension

x = torch.randn((batch_size, s_len, feature_dim)).to(device)
x_mark_enc = torch.randn((batch_size, s_len, 3)).to(device)  # Here 3 is an example, adjust as needed
x_dec = torch.randn((batch_size, 2 * pre_len, feature_dim)).to(device)  # Here 2*pre_len is an example, adjust as needed
x_mark_dec = torch.randn((batch_size, 2 * pre_len, 3)).to(device)  # Here 3 is an example, adjust as needed

# Compute FLOPs
macs, params = profile(model, inputs=(x, x_mark_enc, x_dec, x_mark_dec))
print(f"Total FLOPs: {macs}")
