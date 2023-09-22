import torch
import numpy as np
import random
import matplotlib.pyplot as plt

model_A = Forward_net()
# checkpoint = torch.load('D:/ckpt_parameters/Forward/model0_Fixed_layer_epoch190.pt')
checkpoint = torch.load('D:/ckpt_parameters/Forward/model0_single_layer_epoch1000.pt')
model_A.load_state_dict(checkpoint['model_state_dict'])

model_A.eval()
model_A = model_A.cuda()

# desired_electric_field = 2*np.load('D:/DL_numpy_data/CF_example_2.npy')
# desired_electric_field[0,:,:] = 3*np.load('D:/DL_numpy_data/CF_example_gauss2.npy')[0,:,:]
# desired_electric_field[1,:,:] = 3*np.load('D:/DL_numpy_data/CF_example_gauss2.npy')[1,:,:]
desired_electric_field = 1.5*np.load('D:/DL_numpy_data/CF_example_gauss2.npy')
for i in range(3):
    k = i+1
    plt.figure(figsize=(5, 15))
    plt.subplot(1, 3, k)
    plt.imshow(desired_electric_field[i,:,:], cmap='viridis')
    plt.title(f'Electric Field {i+1}')
    plt.axis('off')
    plt.colorbar()
plt.show()
desired_electric_field = torch.tensor(desired_electric_field).unsqueeze(dim=0)
# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The model, desired output, and unpooling layer are also moved to the device
model_A = model_A.to(device)
desired_electric_field = desired_electric_field.to(device)
unpool = nn.Upsample(scale_factor=2, mode='nearest').to(device)

# Define the size of the structure matrix
MATRIX_SIZE = 16


# Initialize the population with symmetric matrices
def initialize_population(pop_size):
    A = [[0., 0., 1., 1., 1., 0., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1.],
        [0., 0., 1., 1., 0., 1., 1., 0., 1., 1., 1., 1., 1., 0., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 0., 0., 1., 1., 1., 1.],
        [1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 1., 1.],
        [1., 0., 1., 0., 1., 1., 0., 1., 0., 1., 0., 0., 1., 0., 1., 1.],
        [0., 1., 1., 0., 1., 0., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1.],
        [0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 1., 1., 1., 1.],
        [1., 1., 1., 1., 0., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [0., 1., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 0., 1., 1., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]
    
    A = torch.tensor(A).float()
#     A = -A + 1
    population = 500*[A]
#     for _ in range(pop_size-5):
#         half_matrix = torch.randint(0, 2, (MATRIX_SIZE, MATRIX_SIZE))
#         for i in range(MATRIX_SIZE):
#             for j in range(i+1, MATRIX_SIZE):
#                 half_matrix[i][j] = half_matrix[j][i]
#         population.append(half_matrix)
    return population

# Mutate an individual while keeping it symmetric
def mutate(individual, mutation_rate):
    for i in range(MATRIX_SIZE):
        for j in range(MATRIX_SIZE // 2):
            if random.random() < mutation_rate:
                individual[i][j] = 1 - individual[i][j]
                individual[j][i] = individual[i][j]
    return individual

# Crossover two parents to produce two children, ensuring symmetry
def crossover(parent1, parent2):
    child1 = parent1.clone()
    child2 = parent2.clone()
    for i in range(MATRIX_SIZE):
        for j in range(MATRIX_SIZE // 2):
            if random.random() > 0.5:
                child1[i][j], child2[i][j] = child2[i][j], child1[i][j]
                child1[j][i], child2[j][i] = child2[j][i], child1[j][i]
    return child1, child2

def pearson_correlation(x, y):
    """
    Compute the Pearson Correlation Coefficient between two 3x32x32 tensors.
    """
    # Ensure the tensors are on the same device
    if x.device != y.device:
        raise ValueError("Both tensors should be on the same device")

    # Flatten the tensors
    x = x.view(-1)
    y = y.view(-1)

    # Calculate means
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)

    # Calculate numerator: covariance between x and y
    numerator = torch.sum((x - mean_x) * (y - mean_y))

    # Calculate denominators: standard deviations of x and y
    denominator_x = torch.sqrt(torch.sum((x - mean_x) ** 2))
    denominator_y = torch.sqrt(torch.sum((y - mean_y) ** 2))

    # Calculate Pearson Correlation Coefficient
    pearson_corr = numerator / (denominator_x * denominator_y)

    return pearson_corr

def mirror_padding(X):
    X_flip0 = X[:,::-1]
    M_size = np.shape(X)[0]
    X = np.concatenate((X_flip0,X,X_flip0),axis=1)
    X = X[:,M_size//2:]
    X = X[:,:-M_size//2]
    
    X_flip1 = X[::-1,:]
    X_v0 = X_flip1[M_size//2:,:]
    X_v1 = X_flip1[:M_size//2,:]
    X = np.concatenate((X_v0,X,X_v1),axis=0)

    return X

# Define the fitness function based on the difference from the desired electric field
def fitness(individual):
    
# mirror padding
    individual = np.array(individual)
    individual = mirror_padding(individual)
    individual = torch.tensor(individual)
    individual = individual.float().unsqueeze(dim=0).unsqueeze(dim=0)  # Convert to float and move to device
    individual = individual.to(device)
    
#     individual = unpool(individual)
    output = model_A(individual)

    loss01 = 1/F.mse_loss(output[:,0,:,:], desired_electric_field[:,0,:,:]).item()
    loss02 = 1/F.mse_loss(output[:,1,:,:], desired_electric_field[:,1,:,:]).item()
    loss03 = 1/F.mse_loss(output[:,2,:,:], desired_electric_field[:,2,:,:]).item()

    loss1 = pearson_correlation(output[:,0,:,:], desired_electric_field[:,0,:,:]).item()
    loss2 = pearson_correlation(output[:,1,:,:], desired_electric_field[:,1,:,:]).item()
    loss3 = pearson_correlation(output[:,2,:,:], desired_electric_field[:,2,:,:]).item()

#     loss = 1*loss1*loss01 + 2*loss2*loss02 + 1.3*loss3*loss03
    loss = loss01*loss02*loss03*loss1*loss2*loss3
    return loss  # Turn into a maximization problem

def plot_structure_and_fields(structure, electric_fields):
    # Plot structure
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(structure.cpu().numpy(), cmap='gray')
    plt.title('Structure')
    plt.axis('off')
    print(structure[8:24,8:24].cpu())
    # Plot electric fields for each wavelength
    for i in range(3):
        plt.subplot(1, 4, i+2)
        plt.imshow(electric_fields[i,:,:].cpu().numpy(), cmap='viridis')
        plt.title(f'Electric Field {i+1}')
        plt.axis('off')
        plt.colorbar()
    plt.show()
    
# Genetic algorithm parameters
POPULATION_SIZE = 10000
MUTATION_RATE = 0.05
NUM_GENERATIONS = 1000
NUM_ELITE = 500

# Initialize the population
# population = initialize_population(POPULATION_SIZE)
population = elites

# Store average fitness over generations
avg_fitnesses = []
max_fitnesses = []

# Main loop of the genetic algorithm
for generation in tqdm(range(NUM_GENERATIONS)):
    # Evaluate the fitness of each individual in the population
    fitnesses = [fitness(ind) for ind in population]

    # Store average fitness for this generation
    avg_fitness = sum(fitnesses) / len(fitnesses)
    avg_fitnesses.append(avg_fitness)
    
    # Select the top individuals (elites) to be parents for the next generation
    sorted_indices = np.argsort(fitnesses)[::-1]
    elites = [population[i] for i in sorted_indices[:NUM_ELITE]]
    
    max_fitness = np.max(fitnesses)
    max_fitnesses.append(max_fitness)
    
    # Produce the next generation through crossover and mutation
    new_population = []
    for _ in range(POPULATION_SIZE // 2):
        parent1, parent2 = random.choices(elites, k=2)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1, MUTATION_RATE)
        child2 = mutate(child2, MUTATION_RATE)
        new_population.extend([child1, child2])
    population = new_population
    MUTATION_RATE = 0.9995*MUTATION_RATE
    
    # Plot the average fitness over generations
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(avg_fitnesses)
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.title("Average Fitness over Generations")
    
    plt.subplot(1, 2, 2)
    plt.plot(max_fitnesses)
    plt.xlabel("Generation")
    plt.ylabel("Max Fitness")
    plt.title("Max Fitness over Generations")
    plt.show()
    
    example_structure = elites[0].view(MATRIX_SIZE, MATRIX_SIZE).float()
    example_structure = np.array(example_structure)
    example_structure = mirror_padding(example_structure)
    example_structure = torch.tensor(example_structure).view(1,MATRIX_SIZE*2, MATRIX_SIZE*2)
    with torch.no_grad():

        example_structure = example_structure.unsqueeze(0)
#         example_structure = unpool(example_structure)
        electric_fields_output = model_A(example_structure.to(device))

        example_structure = example_structure.squeeze()
        electric_fields_output = electric_fields_output.squeeze()
        print(electric_fields_output.size())
    plot_structure_and_fields(example_structure, electric_fields_output)
    #280 best
