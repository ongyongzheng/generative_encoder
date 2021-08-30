for i in {0..1310}
do
    python test_digital_rock_images.py generate_digital_rock_images 512_generator_with_ae -digital_rock_position $i -with_ae true -gan_name digital_rock_train_pgan_clean -ae_name digital_rock_train_vae_clean
done