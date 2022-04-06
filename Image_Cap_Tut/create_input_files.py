from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                    #    karpathy_json_path='../caption data/dataset_coco.json',
                       karpathy_json_path='../data/caption_datasets/dataset_coco.json',
                    #    image_folder='/media/ssd/caption data/',
                       image_folder='../data/',
                       captions_per_image=5,
                       min_word_freq=5,
                    #    output_folder='/media/ssd/caption data/',
                       output_folder='../data/cap_data/',
                       max_len=50)
