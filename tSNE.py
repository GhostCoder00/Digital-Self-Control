import numpy  as np
import matplotlib.pyplot as plt
import wandb
from sklearn.manifold import TSNE

# ignore tSNE crash (RuntimeWarning: invalid value encountered in divide X_embedded = X_embedded / np.std(X_embedded[:, 0]) * 1e-4)
np.seterr(divide='ignore', invalid='ignore')

def tsne_visualization(features, labels, phase, wandb_log, n_components=2):
    if n_components == 2:
        if phase != '':
            tsne = TSNE(n_components=2, random_state=0)
            x_2d = tsne.fit_transform(features)
            target_ids = range(features.shape[0])
            #fig, ax = plt.subplots()
            #ax.scatter(x_2d[:,0], x_2d[:,1], alpha=0.3)

            fig = plt.figure(figsize=(6, 5))
            colors = 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
            for i, c, label in zip(target_ids, colors, labels):
                plt.scatter(x_2d[labels == i, 0], x_2d[labels == i, 1], c=c, alpha=0.3)
            fig.canvas.draw()
            #plt.savefig(phase[:-1] +'2d.png', bbox_inches='tight')
            image_tsne = np.array(fig.canvas.renderer.buffer_rgba())
            image_tsne = wandb.Image(image_tsne, caption="t-SNE of " + phase)
            # wandb.log({"t-SNE_2d/"+phase[:-1]: image_tsne})
            wandb_log["t-SNE_2d/"+phase[:-1]] = image_tsne

    elif n_components == 3:
        if phase != '':
            tsne = TSNE(n_components=3, random_state=0)
            x_3d = tsne.fit_transform(features)
            target_ids = range(features.shape[0])
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(projection='3d')
            #ax.scatter(x_3d[:,0], x_3d[:,1], x_3d[:,2], alpha=0.3)

            colors = 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
            for i, c, label in zip(target_ids, colors, labels):
                plt.scatter(x_3d[labels == i, 0], x_3d[labels == i, 1], x_3d[labels == i, 2], c=c, alpha=0.3)
            fig.canvas.draw()
            #plt.savefig(phase[:-1] +'3d.png', bbox_inches='tight')
            image_tsne = np.array(fig.canvas.renderer.buffer_rgba())
            image_tsne = wandb.Image(image_tsne, caption="t-SNE of " + phase)
            # wandb.log({"t-SNE_3d/"+phase[:-1]: image_tsne})
            wandb_log["t-SNE_3d/"+phase[:-1]] = image_tsne