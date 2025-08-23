import torch

def get_label_indices(label_windows):
        unique_labels = torch.unique(label_windows)
        label_indices = {}
        for label in unique_labels:
            indices = torch.nonzero(label_windows == label, as_tuple=True)[0]
            label_indices[label.item()] = indices
        return label_indices

def indices_to_replace(real_label_indices, fake_label_indices, proportion):
        indices_to_replace_real = []
        indices_to_replace_fake = []

        for label, real_indices in real_label_indices.items():

            num_indices_to_replace = int(len(real_indices) * proportion)
            fake_indices = fake_label_indices[label]

            if fake_indices.shape[0] >= num_indices_to_replace:
                random_indices_from_real = torch.randperm(
                    real_indices.shape[0])[:num_indices_to_replace]
                random_indices_from_fake = torch.randperm(fake_indices.shape[0])[
                    :num_indices_to_replace]
                
                indices_to_replace_real.append(real_indices[random_indices_from_real])
                indices_to_replace_fake.append(fake_indices[random_indices_from_fake])
            else:
                num_indices_to_replace = fake_indices.shape[0]
                random_indices_from_real = torch.randperm(
                    real_indices.shape[0])[:num_indices_to_replace]
                
                indices_to_replace_real.append(real_indices[random_indices_from_real])
                indices_to_replace_fake.append(fake_indices)
        
        indices_to_replace_real = torch.cat(indices_to_replace_real)
        indices_to_replace_fake = torch.cat(indices_to_replace_fake)
        print(indices_to_replace_real)
        print(indices_to_replace_fake)

        indices_to_replace = torch.stack((indices_to_replace_real, indices_to_replace_fake))
        return indices_to_replace


real_label = torch.tensor([1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2])
fake_label = torch.tensor([2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2])

real_label_indices = get_label_indices(real_label)
fake_label_indices = get_label_indices(fake_label)
print(real_label_indices)
print(fake_label_indices)

proportion = 0.5

indices_to_replace = indices_to_replace(real_label_indices, fake_label_indices, proportion)
print(indices_to_replace)
for pair in indices_to_replace:
    print(pair)