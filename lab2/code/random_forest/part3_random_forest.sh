# 1) First extract AE latent vectors for the 3 labeled images
python ../extract_part3_latent_vectors.py \
  ../configs/finetune_final.yaml \
  ../checkpoints/finetune/final/final-epoch=004.ckpt \
  ../results/part3_latent_vectors.npz

# 2) Train/evaluate grouped Random Forest models
python part3_random_forest.py \
  --ae-features ../results/part3_latent_vectors.npz \
  --labeled-paths \
    ../../image_data_float32/O013257.npz \
    ../../image_data_float32/O013490.npz \
    ../../image_data_float32/O012791.npz \
  --outdir ../results/part3_random_forest

# 3) Optional sanity check on unlabeled images
# If the chosen best model uses AE features, first create unlabeled AE feature npz
# with a script analogous to extract_part3_latent_vectors.py, then run:
python part3_rf_unlabeled_inference.py \
  --model ../results/part3_random_forest/best_random_forest.joblib \
  --metadata ../results/part3_random_forest/best_model_metadata.json \
  --unlabeled-paths ../../image_data_float32/O000001.npz ../image_data_float32/O000002.npz \
  --outdir ../results/part3_random_forest/unlabeled_predictions