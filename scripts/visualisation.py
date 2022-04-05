import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')

def plot_latentSpace(latentSpace, title="UMAP projection of latent space"):
    import umap.umap_ as umap
    
    plt.rcParams.update({'font.size': 18})
    reducer = umap.UMAP()
    print("Doing embedding...")
    embedding = reducer.fit_transform(latentSpace.detach().numpy())
    print(f"UMAP dimension: {embedding.shape}")

    fig, ax = plt.subplots(figsize=(8,8))
    plt.scatter(embedding[:, 0], embedding[:, 1])
    plt.gca().set_aspect('equal', 'datalim')
    #plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title(title, fontsize=18)
    plt.show()

def plot_cpg_reconstruction(model, data_tensor, title="Reconstruction of random CpGs - Test Dataset", size=50):
    plt.rcParams.update({'font.size': 18})
    
    model.eval()
    orig = data_tensor.cpu().detach().numpy()
    recon = model(data_tensor)
    # check if VAE or AE was used
    if isinstance(recon, tuple):
        recon = recon[0].cpu().detach().numpy()
    else:
        recon = recon.detach().numpy()
    r2 = []
    fig, ax = plt.subplots(figsize=(6,6))
    for i in range(size):
        ax.scatter(orig[:,i], recon[:,i], s=8)
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    ax.set_ylabel("Reconstructed beta value")
    ax.set_xlabel("Original beta value")
    ax.set_title(title)
    plt.show()
        