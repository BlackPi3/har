import torch
from torch.utils.data import Dataset


class AdvDataset(Dataset):
    def __init__(self, mhad_accel_windows, mhad_skel_windows, mhad_label_windows, ntu_skel_windows, ntu_label_windows):

        self.accel_windows = mhad_accel_windows
        self.label_windows = mhad_label_windows

        self.propotion = 0.5

        # step 1: see the map where each label is present in the data
        mhad_label_indices = self.get_label_indices(mhad_label_windows)
        ntu_label_indices = self.get_label_indices(ntu_label_windows)

        # step 2: find out which indices to replace
        # shape (2, N) where first row is real indices and second row is fake indices
        indices_to_replace = self.indices_to_replace(mhad_label_indices, ntu_label_indices, self.propotion)

        # step 3: replace the data
        self.adv_skel_windows = self.replace_data(indices_to_replace, mhad_skel_windows, ntu_skel_windows)

        assert len(self.adv_skel_windows) == len(mhad_accel_windows) == len(mhad_label_windows)

        # step 4: set real fake labels
        self.real_fake_labels = torch.ones(len(mhad_label_windows), dtype=torch.float32) # BCELoss expects float not long
        # self.real_fake_labels[indices_to_replace[0]] = 0 # the ones that are being replaced are fake

    def get_label_indices(self, label_windows):
        '''
        label_windows: tensor of shape (N,)
        return dictionary of label to indices. to find out which labels are present where in the data.
        '''
        unique_labels = torch.unique(label_windows)
        label_indices = {}
        for label in unique_labels:
            indices = torch.nonzero(label_windows == label, as_tuple=True)[0]
            label_indices[label.item()] = indices
        return label_indices

    def indices_to_replace(self, real_label_indices, fake_label_indices, proportion):
        '''
        real_label_indices: dict of label to indices
        portion: proportion of real data to replace
        return indices to replace shape (2, num_indices_to_replace) where first row is real indices and second row is fake indices
        '''
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

        indices_to_replace = torch.stack((indices_to_replace_real, indices_to_replace_fake)) # (2, N)
        return indices_to_replace
    
    def replace_data(self, indices_to_replace, real_skel, fake_skel):
        '''
        indices_to_replace: indices to replace shape (2, num_indices_to_replace) where first row is real indices and second row is fake indices
        real_skel: real data
        fake_skel: fake data
        '''
        real_indices = indices_to_replace[0]
        fake_indices = indices_to_replace[1]

        adv_skel = real_skel.clone()
        adv_skel[real_indices] = fake_skel[fake_indices]
        return adv_skel

    def __len__(self):
        return len(self.adv_skel_windows)

    def __getitem__(self, i):
        return self.accel_windows[i], self.adv_skel_windows[i], self.label_windows[i]#, self.real_fake_labels[i]
