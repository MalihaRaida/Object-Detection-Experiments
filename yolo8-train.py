
from ultralytics import YOLO
import torch
import copy


def compare_dicts(state_dict1, state_dict2):
    # Compare the keys
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    if keys1 != keys2:
        print("Models have different parameter names.")
        return False

    # Compare the values (weights)
    for key in keys1:
        if not torch.equal(state_dict1[key], state_dict2[key]):
            print(f"Weights for parameter '{key}' are different.")
            if "bn" in key and "22" not in key:
              state_dict1[key] = state_dict2[key]


def put_in_eval_mode(trainer, n_layers=22):
  for i, (name, module) in enumerate(trainer.model.named_modules()):
    if name.endswith("bn") and int(name.split('.')[1]) < n_layers:
      module.eval()
      module.track_running_stats = False
      # print(name, " put in eval mode.")

# Initialize pretrained model
model = YOLO("yolov8n.pt")
old_dict = copy.deepcopy(model.state_dict())


model.state_dict().keys()



# model.add_callback("on_train_epoch_start", put_in_eval_mode)

# # Train the model. Freeze the first 22 layers [0-21].
# results = model.train(data='dataset.yaml', freeze=22, epochs=100, imgsz=640)

# Compare the dicts. Changes should only be in layer 22 and above
compare_dicts(old_dict, model.state_dict())

new_state_dict = dict()

#  Increment the head number by 1 in the state_dict
for k, v in model.state_dict().items():
  if k.startswith("model.model.22"):
    new_state_dict[k.replace("model.22", "model.23")] = v

# Save the current state_dict. Only layer 23.
torch.save(new_state_dict, "yolov8n_lp.pth")