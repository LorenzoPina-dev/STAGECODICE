import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from gestioneCNN.CustomAdamW import CustomAdamW
import copy
#from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

class ManageCNN:
    def __init__(self, device, model, train_loader, test_loader, lr=1e-3, wd=1e-4):
        self.device = device
        self.model=model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epoch=0
        #self.scaler = GradScaler('dml')
        self.optimizer = CustomAdamW(model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=20, factor=0.5)
        self.patient=50


    def learn(self, num_epochs, verbose=True):
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        acc_train, acc_test = [], []
        self.best_acc=2
        self.epochs_no_improve=0
        for epoch in range(num_epochs):
            loss = self._train_epoch(criterion)
            precTrain=self._get_accuracy(train=True)
            precTest=self._get_accuracy(train=False)
            acc_train.append(precTrain)
            acc_test.append(precTest)
            test_loss=self._get_validation_loss(criterion)
            if self.checkOverfit(test_loss):
                return acc_train, acc_test
            self.epoch+=1
            if verbose:
                print(f'Epoch {epoch+1}/{num_epochs} | Loss: {loss:.4f} | '
                      f'Train Acc: {acc_train[-1]:.2f}% | Test Acc: {acc_test[-1]:.2f}% |test loss: {test_loss:.4f}')
        return acc_train, acc_test
    
    def _get_validation_loss(self, criterion):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                outputs = self.model(inputs)
                if isinstance(outputs, dict) and 'out' in outputs:
                    outputs = outputs['out']
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(self.test_loader)


    def _train_epoch(self, criterion):
        self.model.train()
        total_loss = 0.0
        batch_idx=0
        '''with profile(
            activities=[ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=tensorboard_trace_handler("D:/stage/log"),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        ) as prof:
        '''

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            if isinstance(outputs, dict) and 'out' in outputs:
                outputs = outputs['out']
            loss = criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            #self.scaler.scale(loss).backward()
            #self.scaler.step(self.optimizer)
            #self.scaler.update()
            total_loss += loss.item()
            batch_idx+=1
            #prof.step()

            #if batch_idx > 5:
            #    break 

        return total_loss / len(self.train_loader)

    def show_segmentations(self, num_images=3):
        self.model.eval()
        images_shown = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                if isinstance(outputs, dict) and 'out' in outputs:
                    outputs = outputs['out']
                
                preds = torch.argmax(outputs, dim=1)

                for i in range(min(num_images, inputs.size(0))):
                    img = inputs[i].cpu()
                    pred = preds[i].cpu()
                    label = labels[i].cpu()

                    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                    axs[0].imshow(F.to_pil_image(img))
                    axs[0].set_title("Input")
                    axs[1].imshow(pred, cmap="jet")
                    axs[1].set_title("Predizione")
                    axs[2].imshow(label, cmap="jet")
                    axs[2].set_title("Ground Truth")
                    for ax in axs:
                        ax.axis('off')
                    plt.tight_layout()
                    plt.show()

                    images_shown += 1
                    if images_shown >= num_images:
                        return

    def checkOverfit(self,acc):
            # Scheduler (val_loss opzionale se usi ReduceLROnPlateau)
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(1 - acc)  # loss proxy
        else:
            self.scheduler.step()

        # --- EARLY STOPPING ---
        if acc < self.best_acc:
            self.best_acc = acc
            best_model_wts = copy.deepcopy(self.model.state_dict())
            self.epochs_no_improve = 0
            torch.save(self.model.state_dict(), "model")
            print(f"[INFO] Best model saved with acc: {self.best_acc:.4f}")
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patient:
                print("[INFO] Early stopping")
                return True
        return False

    def _get_accuracy(self, train=True):
        self.model.eval()
        loader = self.train_loader if train else self.test_loader
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                outputs = self.model(inputs)
                if isinstance(outputs, dict) and 'out' in outputs:
                    outputs = outputs['out']
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
        return 100 * correct / total
    
    def save(self, path):
        """
        Save only the training state (epoch, model weights, optimizer state) to disk.
        DataLoaders are recreated externally when loading.
        """
        print("INIZIO SALVATAGGIO")
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        # Debug prints
        print("Saving checkpoint with:")
        for k, v in checkpoint.items():
            print(f"  {k}: {type(v)}")

        torch.save(checkpoint, path)

    def load(path, device, model, train_loader, test_loader, lr=1e-3, wd=1e-4):
        """
        Load model and optimizer state from disk and return a ManageCNN instance.
        """
        # Initialize manager
        manager = ManageCNN(device, model, train_loader, test_loader, lr=lr, wd=wd)
        # Load checkpoint
        checkpoint = torch.load(path, map_location=device)
        manager.model.load_state_dict(checkpoint['model_state_dict'])
        manager.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        manager.start_epoch = checkpoint.get('epoch', 0)
        print(f"Checkpoint loaded from {path}, starting at epoch {manager.start_epoch}")
        return manager
    
    def get_predictions(self, train=False):
        """
        Compute and return true labels and predictions.

        Args:
            train (bool): if True, use train_loader; otherwise use test_loader.

        Returns:
            y_true (List[int]): ground-truth labels
            y_pred (List[int]): predicted labels
        """
        self.model.eval()
        loader = self.train_loader if train else self.test_loader
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                if isinstance(outputs, dict) and 'out' in outputs:
                    outputs = outputs['out']
                _, preds = torch.max(outputs, dim=1)
                y_pred.extend(preds.cpu().tolist())
                y_true.extend(labels.cpu().tolist())

        return y_true, y_pred