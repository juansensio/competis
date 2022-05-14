from src import AllDataModule
from tqdm import tqdm
import torch
from src import AllModule

state_dict = torch.load(
    'checkpoints/all-val_error=0.70873-epoch=3.ckpt')['state_dict']
model = AllModule(dict(backbone='resnet18', pretrained=True,
                  mlp_layers=[256, 512], mlp_dropout=0.))
model.load_state_dict(state_dict)
model.cuda()

tta_trans = [
    None,
    {'HorizontalFlip': {'p': 1}},
    {'VerticalFlip': {'p': 1}},
    {'Transpose': {'p': 1}},
    {'RandomRotate90': {'p': 1}},
]

all_labels = []
for r, trans in enumerate(tta_trans):
    dm = AllDataModule(batch_size=512, num_workers=10,
                       pin_memory=True, test_trans=trans)
    dm.setup()
    labels, observations = [], []
    for batch in tqdm(dm.test_dataloader()):
        preds = model.predict(batch)
        values, ixs = preds.topk(30)
        labels += [' '.join([str(i.item()) for i in ix]) for ix in ixs]
        observation_ids = batch['observation_id']
        observations += observation_ids.cpu().numpy().tolist()
    all_labels.append(labels)
