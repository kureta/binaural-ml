import torch
from your_rnn_model import StreamingTrajectoryDecoderRNN

model = StreamingTrajectoryDecoderRNN(hidden_dim=128, num_layers=1)
model.load_state_dict(torch.load("best_rnn_model.pth"))

model = model.eval().cuda()

# Export TorchScript
example_input = (torch.randn(1, 960, 2).cuda(), torch.randn(1, 1, 128).cuda())
scripted_model = torch.jit.trace(model, example_input)

scripted_model.save("server_inference/model/streaming_traj_decoder.pt")
print("TorchScript model saved.")
