# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:24:53 2022

@author: SHM
"""
def plot_history(net_history):
    import matplotlib.pyplot as plt
    history = net_history.history
    losses = history['loss']
    val_losses = history['val_loss']
    accuracies = history['accuracy']
    val_accuracies = history['val_accuracy']
    plt.figure()  
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses, 'red')
    plt.grid()
    plt.plot(val_losses, 'blue')
    plt.legend(['loss', 'val_loss'])    
    plt.figure()
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.plot(accuracies)
    plt.plot(val_accuracies)
    plt.legend(['accuracy', 'val_accuracy'])