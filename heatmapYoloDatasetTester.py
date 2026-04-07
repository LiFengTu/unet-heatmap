import myDataset

#unetHeatmapYoloDataset.ResetDataFolder('data')
dataset = myDataset.HeatmapYOLODataset(
        data_dir="data",
        stride=4,
        sigma_factor=0.3,
        input_size=1024,
        phase='val'
    )
    
sample = dataset[0]
print("Image shape:", sample['image'].shape)
print("Heatmap shape:", sample['heatmap'].shape)
print("Number of objects:", sample['num_objects'])