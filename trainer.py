import torch
from tqdm import tqdm

def train(model, loader, optimizer, criterion, metric, device):
	epoch_loss = 0
	epoch_acc = 0

	model.train()

	for (item1, item2), target in loader:
		item1 = item1.to(device)
		item2 = item2.to(device)
		target = target.to(device)

		optimizer.zero_grad()

		outputs = model(item1, item2)
		loss = criterion(*outputs, target)
		acc = metric(*outputs, target)

		loss.backward()
		optimizer.step()

		epoch_loss += loss.item()
		epoch_acc += acc.item()

	return epoch_loss / len(loader), epoch_acc / len(loader)


def evaluate(model, loader, criterion, metric, device):
	epoch_loss = 0
	epoch_acc = 0

	model.eval()

	with torch.no_grad():
		for (item1, item2), target in loader:
			item1 = item1.to(device)
			item2 = item2.to(device)
			target = target.to(device)

			outputs = model(item1, item2)
			loss = criterion(*outputs, target)
			acc = metric(*outputs, target)

			epoch_loss += loss.item()
			epoch_acc += acc.item()

	return epoch_loss / len(loader), epoch_acc / len(loader)


def test(model, loader, metric, device):
	epoch_acc = 0
	with torch.no_grad():
		for (item1, item2) , target in loader:
			item1 = item1.to(device)
			item2 = item2.to(device)
			target = target.to(device)

			outputs = model(item1, item2)
			acc = metric(*outputs, target)
			epoch_acc += acc.item()

	return epoch_acc / len(loader)


def fit(model, loaders, optimizer, criterion, metric, writer, save_path, config):
	best_valid_loss = float('inf')

	for epoch in tqdm(range(config['num_epochs'])):

		train_loss, train_acc = train(model, loaders['train'], optimizer, criterion, metric, config['device'])
		valid_loss, valid_acc = evaluate(model, loaders['valid'], criterion, metric, config['device'])

		if valid_loss < best_valid_loss:
			best_valid_loss = valid_loss
			torch.save(model.state_dict(), save_path)

		# log the running loss
		writer.add_scalar('Loss/train', train_loss, epoch)
		writer.add_scalar('Loss/validation', valid_loss, epoch)
		writer.add_scalar('Accuracy/train', train_acc, epoch)
		writer.add_scalar('Accuracy/validation', valid_acc, epoch)