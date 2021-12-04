# Face-Generation

On the CelebA dataset, I defined and trained a DCGAN in this research.
This project's purpose is to create a generator network that can generate fresh images of faces that are as realistic as possible.

### Architecture :

- 4-Layer CNN Generator - Given a latent vector z, creates a new face picture using learnt weights from photos in the training set.
- Discriminator: 5-Layer CNN - Determines whether a face image is real or fake.
It attempts to persuade the Dircriminator that the created image is genuine.
  
### Hyperparameters :
num epochs = 15 
- learning rate = 0.001 with Adam optimizer: beta1=0.1,beta2=0.999
- z = 100 is the length of the latent vector.
- added batch normalisation 
- number of filters in Discriminator's first hidden layer = 32 
- number of filters in Generator's first hidden layer = 32
  
### Training :
It was necessary to alternate between training the discriminator and the generator during the training process.
- 15 epochs
