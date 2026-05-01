import os
import time
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class MetricLogger:
    def __init__(self, log_dir, class_names, model_name, class_weights=None, val_sample_weights=None):
        self.log_dir = os.path.join("logs", log_dir, model_name, time.strftime("%Y%m%d-%H%M%S"))
        self.writer = SummaryWriter(self.log_dir)
        self.class_names = class_names
        self.nc_equivalent_indices = self._resolve_class_indices(('c', 'n'))
        self.model_size = None
        self.flops = None
        if class_weights is not None and hasattr(class_weights, 'detach'):
            class_weights = class_weights.detach().cpu().numpy()
        if val_sample_weights is not None and hasattr(val_sample_weights, 'detach'):
            val_sample_weights = val_sample_weights.detach().cpu().numpy()
        self.class_weights = None if class_weights is None else np.asarray(class_weights, dtype=np.float32)
        self.val_sample_weights = None if val_sample_weights is None else np.asarray(val_sample_weights, dtype=np.float32)
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        self.val_metrics = {'loss': [], 'outputs': [], 'labels': [], 'accuracy': 0.0}
        self.epoch = 0  # 添加 epoch 属性
        self.reset()

    def _resolve_class_indices(self, class_names):
        index_map = {name: index for index, name in enumerate(self.class_names)}
        if not all(name in index_map for name in class_names):
            return None
        return np.asarray([index_map[name] for name in class_names], dtype=np.int64)

    def reset(self):
        self.train_metrics = {'loss': [], 'outputs': [], 'labels': []}
        self.val_metrics = {'loss': [], 'outputs': [], 'labels': [], 'accuracy': 0.0, 'accuracy_nc_merged': 0.0}

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

    def _get_accuracy_weights(self, phase, labels):
        if phase == 'val' and self.val_sample_weights is not None:
            return self.val_sample_weights
        if phase == 'val' and self.class_weights is not None:
            return self.class_weights[labels.astype(np.int64)]
        return None

    def _compute_accuracy(self, matches, weights=None):
        if weights is None:
            return float(np.mean(matches))
        return float(np.average(matches, weights=weights))

    def _compute_nc_merged_accuracy(self, preds, labels, weights=None):
        if self.nc_equivalent_indices is None:
            return self._compute_accuracy(preds == labels, weights)

        nc_match = np.isin(labels, self.nc_equivalent_indices) & np.isin(preds, self.nc_equivalent_indices)
        merged_matches = (preds == labels) | nc_match
        return self._compute_accuracy(merged_matches, weights)

    def _calculate_metrics(self, phase):
        outputs = np.concatenate(self.__dict__[f'{phase}_metrics']['outputs'])
        labels = np.concatenate(self.__dict__[f'{phase}_metrics']['labels'])
        preds = np.argmax(outputs, axis=1)

        # 基础指标
        weights = self._get_accuracy_weights(phase, labels)
        accuracy = self._compute_accuracy(preds == labels, weights)
        accuracy_nc_merged = self._compute_nc_merged_accuracy(preds, labels, weights)
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
            'accuracy_nc_merged': accuracy_nc_merged,
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
        self.val_metrics['accuracy_nc_merged'] = val_stats['accuracy_nc_merged']
        self.history['train_loss'].append(train_stats['loss'])
        self.history['train_accuracy'].append(train_stats['accuracy'])
        self.history['val_loss'].append(val_stats['loss'])
        self.history['val_accuracy'].append(val_stats['accuracy'])

        # 记录到TensorBoard
        self._log_to_tensorboard(train_stats, val_stats)
        
        # 生成可视化报告
        self._save_confusion_matrix(val_stats['cm_norm'], 'val_confusion_matrix')
        self._save_classification_report(val_stats['report'], val_stats['accuracy_nc_merged'])
        self._save_classification_metrics_chart(val_stats['report'], 'val_classification_metrics')
        self._save_training_curves('training_curves')
        
        # 打印概要
        print(f"Epoch Summary - Train Loss: {train_stats['loss']:.4f} | Val Loss: {val_stats['loss']:.4f}")
        print(f"Val Accuracy: {val_stats['accuracy']:.2%}")
        print(f"Val Accuracy (c/n merged): {val_stats['accuracy_nc_merged']:.2%}")
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
            self.val_metrics['accuracy_nc_merged'] = stats['val']['accuracy_nc_merged']
        
        # 记录到TensorBoard
        if 'train' in stats and 'val' in stats:
            self._log_to_tensorboard(stats['train'], stats['val'])
        elif 'val' in stats:
            self._log_to_tensorboard({'loss':0, 'accuracy':0}, stats['val'])
        
        # 生成可视化报告
        if 'val' in stats:
            self._save_confusion_matrix(stats['val']['cm_norm'], 'val_confusion_matrix')
            self._save_classification_report(stats['val']['report'], stats['val']['accuracy_nc_merged'])
            self._save_classification_metrics_chart(stats['val']['report'], 'val_classification_metrics')
        
        # 打印结果
        if 'val' in stats:
            print(f"Val Accuracy: {stats['val']['accuracy']:.2%}")
            print(f"Val Accuracy (c/n merged): {stats['val']['accuracy_nc_merged']:.2%}")

    def _log_to_tensorboard(self, train_stats, val_stats):
        self.writer.add_scalars('Loss', {'train': train_stats['loss'], 'val': val_stats['loss']}, self.epoch)
        self.writer.add_scalar('Accuracy/train', train_stats['accuracy'], self.epoch)
        self.writer.add_scalar('Accuracy/val', val_stats['accuracy'], self.epoch)
        self.writer.add_scalar('Accuracy/train_nc_merged', train_stats.get('accuracy_nc_merged', 0.0), self.epoch)
        self.writer.add_scalar('Accuracy/val_nc_merged', val_stats['accuracy_nc_merged'], self.epoch)
        
        # 记录混淆矩阵图像
        fig = self._plot_confusion_matrix(val_stats['cm_norm'])
        self.writer.add_figure('Confusion Matrix', fig, self.epoch)
        plt.close(fig)

        fig = self._plot_classification_metrics(val_stats['report'])
        self.writer.add_figure('Validation/Class Metrics', fig, self.epoch)
        plt.close(fig)

        if self.history['val_loss']:
            fig = self._plot_training_curves()
            self.writer.add_figure('Training/Curves', fig, self.epoch)
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
        fig.savefig(os.path.join(self.log_dir, f'{name}.png'), bbox_inches='tight')
        plt.close(fig)

    def _save_classification_report(self, report, accuracy_nc_merged=None):
        with open(os.path.join(self.log_dir, 'classification_report.txt'), 'w') as f:
            f.write(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10}\n")
            for cls in self.class_names:
                f.write(f"{cls:<10} {report[cls]['precision']:<10.2f} {report[cls]['recall']:<10.2f} {report[cls]['f1-score']:<10.2f}\n")
            f.write(f"\nOverall Accuracy: {report['accuracy']:.2%}")
            if accuracy_nc_merged is not None:
                f.write(f"\nOverall Accuracy (c/n merged): {accuracy_nc_merged:.2%}")

    def _plot_classification_metrics(self, report):
        metrics = ['precision', 'recall', 'f1-score']
        x = np.arange(len(self.class_names))
        width = 0.24

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for index, metric in enumerate(metrics):
            values = [report[class_name][metric] for class_name in self.class_names]
            axes[0].bar(x + (index - 1) * width, values, width=width, label=metric)

        axes[0].set_xticks(x)
        axes[0].set_xticklabels(self.class_names)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel('Score')
        axes[0].set_title('Per-Class Metrics')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.25)

        supports = [report[class_name]['support'] for class_name in self.class_names]
        axes[1].bar(self.class_names, supports, color='#4C78A8')
        axes[1].set_ylabel('Samples')
        axes[1].set_title('Validation Sample Distribution')
        axes[1].grid(axis='y', alpha=0.25)

        fig.suptitle(f"Validation Summary | Accuracy: {report['accuracy']:.2%}")
        fig.tight_layout()
        return fig

    def _save_classification_metrics_chart(self, report, name):
        fig = self._plot_classification_metrics(report)
        fig.savefig(os.path.join(self.log_dir, f'{name}.png'), bbox_inches='tight')
        plt.close(fig)

    def _plot_training_curves(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs = np.arange(1, len(self.history['train_loss']) + 1)

        axes[0].plot(epochs, self.history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(epochs, self.history['val_loss'], label='Val Loss', marker='o')
        axes[0].set_title('Loss Curve')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(alpha=0.25)
        axes[0].legend()

        axes[1].plot(epochs, self.history['train_accuracy'], label='Train Accuracy', marker='o')
        axes[1].plot(epochs, self.history['val_accuracy'], label='Val Accuracy', marker='o')
        axes[1].set_title('Accuracy Curve')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(alpha=0.25)
        axes[1].legend()

        fig.tight_layout()
        return fig

    def _save_training_curves(self, name):
        if not self.history['train_loss']:
            return
        fig = self._plot_training_curves()
        fig.savefig(os.path.join(self.log_dir, f'{name}.png'), bbox_inches='tight')
        plt.close(fig)

    def get_report_paths(self):
        return {
            'log_dir': self.log_dir,
            'classification_report': os.path.join(self.log_dir, 'classification_report.txt'),
            'confusion_matrix': os.path.join(self.log_dir, 'val_confusion_matrix.png'),
            'classification_metrics': os.path.join(self.log_dir, 'val_classification_metrics.png'),
            'training_curves': os.path.join(self.log_dir, 'training_curves.png')
        }
