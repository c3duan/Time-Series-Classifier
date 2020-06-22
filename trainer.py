import torch
import numpy as np
from tqdm import tqdm
from metrics import metric_at_k

def train(model, loader, optimizer, criterion, device):
	epoch_loss = 0

	model.train()
	for (item1, item2), target in loader:
		item1 = item1.to(device)
		item2 = item2.to(device)
		target = target.to(device)

		optimizer.zero_grad()

		outputs = model(item1, item2)
		loss = criterion(*outputs, target)

		loss.backward()
		optimizer.step()

		epoch_loss += loss.item()

	return epoch_loss / len(loader)


def evaluate(model, loader, criterion, device):
	epoch_loss = 0
	model.eval()
	with torch.no_grad():
		for (item1, item2), target in loader:
			item1 = item1.to(device)
			item2 = item2.to(device)
			target = target.to(device)

			outputs = model(item1, item2)
			loss = criterion(*outputs, target)
			epoch_loss += loss.item()

	return epoch_loss / len(loader)


def test(model, loader, criterion, device):
	epoch_loss = 0
	with torch.no_grad():
		for (item1, item2) , target in loader:
			item1 = item1.to(device)
			item2 = item2.to(device)
			target = target.to(device)

			outputs = model(item1, item2)
			loss = criterion(*outputs, target)
			epoch_loss += loss.item()

	return epoch_loss / len(loader)


def fit(model, loaders, optimizer, criterion, writer, save_path, config):
	best_valid_loss = float('inf')

	for epoch in tqdm(range(config['num_epochs'])):

		train_loss = train(model, loaders['train'], optimizer, criterion, config['device'])
		valid_loss = evaluate(model, loaders['valid'], criterion, config['device'])

		queries = torch.cat([
			loaders['all'].dataset.data[
				np.random.choice(loaders['all'].dataset.label_to_indices[label], 1)[0]
			].view(1, -1)
			for label in sorted(loaders['all'].dataset.labels_set)
		], 0).to(config['device'])

		test_acc = metric_at_k(config['top_k'], model, queries, loaders['test'], config['device'])

		if valid_loss < best_valid_loss:
			best_valid_loss = valid_loss
			torch.save(model.state_dict(), save_path)

		# log the running loss
		writer.add_scalar('Loss/train', train_loss, epoch)
		writer.add_scalar('Loss/validation', valid_loss, epoch)
		writer.add_scalar('Accuracy/test', test_acc, epoch)
