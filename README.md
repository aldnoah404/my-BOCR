# Bayesian Optimization with Cluster and Rollback for CNN Pruning
Open source code for our ECCV2022 accepted paper.
https://link.springer.com/chapter/10.1007/978-3-031-20050-2_29

## Use BOCR:
  First, softlink imagenet2012 dataset to ./dataset: 
  
    ln -s PATH_to_your_local_imagenet ./dataset/imagenet
  
  Next, check the environment requirements in requirements.txt
  
  Finally, run the script in the base folder:
  
  For mobilenetv1 experiments:
  
    ./mobilenetv1.sh [options]
      
  For mobilenetv2 experiments:
  
    ./mobilenetv2.sh [options]
      
  For resnet56 experiments:
  
    ./resnet56.sh [options]
      
  Options include non, static, db and gb, each stands for naive BO, layer clustering, direct rollback and gradual rollback.
