import os
import time
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class MetricLogger:
    def __init__(self, log_dir, class_names, model_name, class_weights=None):
        self.log_dir = os.path.join(log_dir, model_name, time.strftime("%Y%m%d-%H%M%S"))
        self.writer = SummaryWriter(self.log_dir)
        self.class_names = class_names
        self.model_size = None
        self.flops = None
        self.class_weights = class_weights
        self.val_metrics = {'loss': [], 'outputs': [], 'labels': [], 'accuracy': 0.0}
        self.epoch = 0  # 添加 epoch 属性
        self.reset()

    def reset(self):
        self.train_metrics = {'loss': [], 'outputs': [], 'labels': []}
        self.val_metrics = {'loss': [], 'outputs': [], 'labels': []}

    def new_epoch(self):
        self.reset()

    def log_train(self, loss, outputs, labels):
        self.train_metrics['loss'].append(loss)
        self.train_metrics['outputs'].append(outputs.detach().cpu().numpy())
        self.train_metrics['labels'].append(labels.detach().cpu().numpy())

    def log_val(self, loss, outputs, labels):
        self.val_metrics['loss'].append(loss)
        self.val_metrics['outputs'].append(outputs.detach().cpu().numpy())
        self.val_metrics['labels'].append(labels.detach().cpu().numpy())

    def _calculate_metrics(self, phase):
        outputs = np.concatenate(self.__dict__[f'{phase}_metrics']['outputs'])
        labels = np.concatenate(self.__dict__[f'{phase}_metrics']['labels'])
        preds = np.argmax(outputs, axis=1)

        # 基础指标
        if self.class_weights is not None:
            accuracy = np.average(preds == labels, weights=self.class_weights)
        else:
            accuracy = np.mean(preds == labels)
        loss = np.mean(self.__dict__[f'{phase}_metrics']['loss'])

        # 混淆矩阵
        cm = confusion_matrix(labels, preds)
        if cm.sum(axis=1)[:, np.newaxis].any():
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        else:
            cm_norm = cm.astype('float')
        
        # 分类报告
        report = classification_report(labels, preds, target_names=self.class_names, output_dict=True, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'loss': loss,
            'cm': cm,
            'cm_norm': cm_norm,
            'report': report
        }
    
    def set_model_size(self, model_size):
        self.model_size = model_size

    def set_flops(self, flops):
        self.flops = flops

    def finalize_epoch(self):
        # 计算指标
        train_stats = self._calculate_metrics('train')
        val_stats = self._calculate_metrics('val')
        self.val_metrics['accuracy'] = val_stats['accuracy']

        # 记录到TensorBoard
        self._log_to_tensorboard(train_stats, val_stats)
        
        # 生成可视化报告
        self._save_confusion_matrix(val_stats['cm_norm'], 'val_confusion_matrix')
        self._save_classification_report(val_stats['report'])
        
        # 打印概要
        print(f"Epoch Summary - Train Loss: {train_stats['loss']:.4f} | Val Loss: {val_stats['loss']:.4f}")
        print(f"Val Accuracy: {val_stats['accuracy']:.2%}")
        if self.model_size is not None:
            print(f"Model Size: {self.model_size:.2f} MB")
        if self.flops is not None:
            print(f"FLOPs: {self.flops:.2f} M")

        self.epoch += 1  # 更新 epoch 计数器

    def finalize_val(self):
        # 只计算存在数据的指标
        stats = {}
        if len(self.val_metrics['outputs']) > 0:
            stats['val'] = self._calculate_metrics('val')
        if len(self.train_metrics['outputs']) > 0:
            stats['train'] = self._calculate_metrics('train')
        
        # 更新验证准确率
        if 'val' in stats:
            self.val_metrics['accuracy'] = stats['val']['accuracy']
        
        # 记录到TensorBoard
        if 'train' in stats and 'val' in stats:
            self._log_to_tensorboard(stats['train'], stats['val'])
        elif 'val' in stats:
            self._log_to_tensorboard({'loss':0, 'accuracy':0}, stats['val'])
        
        # 生成可视化报告
        if 'val' in stats:
            self._save_confusion_matrix(stats['val']['cm_norm'], 'val_confusion_matrix')
            self._save_classification_report(stats['val']['report'])
        
        # 打印结果
        if 'val' in stats:
            print(f"Val Accuracy: {stats['val']['accuracy']:.2%}")

    def _log_to_tensorboard(self, train_stats, val_stats):
        self.writer.add_scalars('Loss', {'train': train_stats['loss'], 'val': val_stats['loss']}, self.epoch)
        self.writer.add_scalar('Accuracy/train', train_stats['accuracy'], self.epoch)
        self.writer.add_scalar('Accuracy/val', val_stats['accuracy'], self.epoch)
        
        # 记录混淆矩阵图像
        fig = self._plot_confusion_matrix(val_stats['cm_norm'])
        self.writer.add_figure('Confusion Matrix', fig, self.epoch)
        plt.close(fig)

    def _plot_confusion_matrix(self, cm):
        fig, ax = plt.subplots(figsize=(8,8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=self.class_names, yticklabels=self.class_names,
               title="Normalized Confusion Matrix",
               ylabel='True label',
               xlabel='Predicted label')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        return fig

    def _save_confusion_matrix(self, cm, name):
        fig = self._plot_confusion_matrix(cm)
        fig.savefig(os.path.join(self.log_dir, f'{name}.png'))
        plt.close(fig)

    def _save_classification_report(self, report):
        with open(os.path.join(self.log_dir, 'classification_report.txt'), 'w') as f:
            f.write(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10}\n")
            for cls in self.class_names:
                f.write(f"{cls:<10} {report[cls]['precision']:<10.2f} {report[cls]['recall']:<10.2f} {report[cls]['f1-score']:<10.2f}\n")
            f.write(f"\nOverall Accuracy: {report['accuracy']:.2%}")
