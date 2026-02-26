#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для генерации диаграмм архитектур нейронных сетей
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Создаем директорию для диаграмм если её нет
os.makedirs('NN_pics', exist_ok=True)

def create_diagram(name, layers, output_path):
    """Создает диаграмму архитектуры нейронной сети"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(layers) + 1)
    ax.axis('off')
    
    # Цвета для разных типов слоев
    colors = {
        'input': '#E8F4F8',
        'embedding': '#B3E5FC',
        'rnn': '#C8E6C9',
        'transformer': '#FFF9C4',
        'conv': '#FFCCBC',
        'dense': '#F8BBD0',
        'output': '#E1BEE7'
    }
    
    y_pos = len(layers)
    for i, layer in enumerate(layers):
        layer_name = layer['name']
        layer_type = layer.get('type', 'dense')
        color = colors.get(layer_type, '#E0E0E0')
        
        # Создаем прямоугольник для слоя
        box = FancyBboxPatch(
            (1, y_pos - 0.3), 8, 0.6,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(box)
        
        # Добавляем текст
        ax.text(5, y_pos, layer_name, 
                ha='center', va='center', 
                fontsize=11, weight='bold')
        
        # Добавляем стрелку вниз (кроме последнего слоя)
        if i < len(layers) - 1:
            arrow = FancyArrowPatch(
                (5, y_pos - 0.3), (5, y_pos - 0.7),
                arrowstyle='->', mutation_scale=20,
                linewidth=2, color='black'
            )
            ax.add_patch(arrow)
        
        y_pos -= 1
    
    # Заголовок
    ax.text(5, len(layers) + 0.5, name, 
            ha='center', va='center', 
            fontsize=16, weight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Создана диаграмма: {output_path}")

# GRU архитектура
gru_layers = [
    {'name': 'Input\n(genome sequence)', 'type': 'input'},
    {'name': 'Embedding\n(NoEmbedding/ConvEmbedding/Embedding)', 'type': 'embedding'},
    {'name': 'Bidirectional GRU\n(hidden_size=32, num_layers=4)', 'type': 'rnn'},
    {'name': 'Predictor\n(3x Linear + Dropout)', 'type': 'dense'},
    {'name': 'Output\n(n_class logits)', 'type': 'output'}
]
create_diagram('GRU Architecture (GruLabeler)', gru_layers, 'NN_pics/GRU/architecture.png')

# LSTM архитектура
lstm_layers = [
    {'name': 'Input\n(genome sequence)', 'type': 'input'},
    {'name': 'ConvEmbedding\n(4 layers, kernel=21)', 'type': 'embedding'},
    {'name': 'Bidirectional LSTM\n(hidden_size=n_class, num_layers=4)', 'type': 'rnn'},
    {'name': 'Predictor\n(3x Linear + Dropout)', 'type': 'dense'},
    {'name': 'Output\n(n_class logits)', 'type': 'output'}
]
create_diagram('LSTM Architecture (LstmLabeler)', lstm_layers, 'NN_pics/LSTM/architecture.png')

# GRU ComDist архитектура
gru_comdist_layers = [
    {'name': 'Input\n(genome sequence)', 'type': 'input'},
    {'name': 'Embedding', 'type': 'embedding'},
    {'name': 'Bidirectional GRU\n(hidden_size, num_layers)', 'type': 'rnn'},
    {'name': 'Predictor\n(3x Linear + Dropout)', 'type': 'dense'},
    {'name': 'Output\n(KLDivLoss)', 'type': 'output'}
]
create_diagram('GRU ComDist Architecture (GruComDistLabeler)', gru_comdist_layers, 'NN_pics/GRU_ComDist/architecture.png')

# GRU Ordinal архитектура
gru_ordinal_layers = [
    {'name': 'Input\n(genome sequence)', 'type': 'input'},
    {'name': 'GruLabeler\n(Embedding + GRU + Predictor)', 'type': 'rnn'},
    {'name': 'OrdinalHead\n(Linear + Bias)', 'type': 'dense'},
    {'name': 'Output\n(ordinal logits)', 'type': 'output'}
]
create_diagram('GRU Ordinal Architecture (GruLabelerOrdinal)', gru_ordinal_layers, 'NN_pics/GRU_Ordinal/architecture.png')

# Simple архитектура
simple_layers = [
    {'name': 'Input\n(genome sequence)', 'type': 'input'},
    {'name': 'WindowSlider\n(multiple window sizes)', 'type': 'conv'},
    {'name': 'Linear Layer\n(projection)', 'type': 'dense'},
    {'name': 'Predictor\n(3x Linear + Dropout)', 'type': 'dense'},
    {'name': 'Output\n(n_class logits)', 'type': 'output'}
]
create_diagram('Simple Architecture (SimpleLabeler)', simple_layers, 'NN_pics/Simple/architecture.png')

# Reformer архитектура
reformer_layers = [
    {'name': 'Input\n(genome sequence)', 'type': 'input'},
    {'name': 'ConvEmbedding\n(4 layers, kernel=21)', 'type': 'embedding'},
    {'name': 'Reformer Transformer\n(LSH + Local Attention, 6 layers)', 'type': 'transformer'},
    {'name': 'Predictor\n(3x Linear + Dropout)', 'type': 'dense'},
    {'name': 'Output\n(n_class logits)', 'type': 'output'}
]
create_diagram('Reformer Architecture (ReformerLabeler)', reformer_layers, 'NN_pics/Reformer/architecture.png')

# SPIDNA архитектура
spidna_layers = [
    {'name': 'Input\n(position + SNP features)', 'type': 'input'},
    {'name': 'Conv2D Position\n(BatchNorm + ReLU)', 'type': 'conv'},
    {'name': 'Conv2D SNP\n(BatchNorm + ReLU)', 'type': 'conv'},
    {'name': 'SPIDNABlock(s)\n(Conv2D + MaxPool + Residual)', 'type': 'conv'},
    {'name': 'FC Layer\n(Linear)', 'type': 'dense'},
    {'name': 'Output\n(num_output)', 'type': 'output'}
]
create_diagram('SPIDNA Architecture', spidna_layers, 'NN_pics/SPIDNA/architecture.png')

# DeepCompressor архитектура
deepcompressor_layers = [
    {'name': 'Input\n(sequence)', 'type': 'input'},
    {'name': 'ConvBlock 1\n(Conv1D + BatchNorm + ReLU + MaxPool)', 'type': 'conv'},
    {'name': 'ConvBlock 2...N\n(multiple conv blocks)', 'type': 'conv'},
    {'name': 'Dense 1\n(Linear + ReLU + Dropout)', 'type': 'dense'},
    {'name': 'Dense 2\n(Linear + ReLU + Dropout)', 'type': 'dense'},
    {'name': 'Dense 3\n(Linear + LogSoftmax)', 'type': 'dense'},
    {'name': 'Output\n(n_output)', 'type': 'output'}
]
create_diagram('DeepCompressor Architecture', deepcompressor_layers, 'NN_pics/DeepCompressor/architecture.png')

print("\nВсе диаграммы успешно созданы!")
