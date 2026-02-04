import pickle
import pandas as pd

class CTPE:
    def __init__(self, name, is_positive, parser, num_slice, first_appear, avg_bbox, last_appear):
        self.study_num = name
        self.is_positive = is_positive
        self.phase = parser
        self.num_slice = num_slice
        self.first_appear = first_appear
        self.bbox = avg_bbox
        self.last_appear = last_appear

    def __len__(self):
        return self.num_slice

# 自动识别路径
if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, default=None)
    parser.add_argument('--output_pkl', type=str, default=None)
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # e:\rerun2
    
    if args.input_csv:
        bbox_path = args.input_csv
    else:
        bbox_path = os.path.join(project_root, 'data', 'G_first_last_nor.csv')
        
    if args.output_pkl:
        pkl_path = args.output_pkl
    else:
        pkl_path = os.path.join(project_root, 'data', 'series_list_last_AG.pkl')

    df = pd.read_csv(bbox_path)
    with open(pkl_path, 'wb') as pkl_file:
        all_ctpes = [CTPE(name = row['NewPatientID'], 
                        is_positive = row['label'],
                        parser = row['parser'],
                        num_slice = row['num_slice'],
                        first_appear = row['first_appear'],
                        avg_bbox = eval(row['avg_bbox']) if isinstance(row['avg_bbox'], str) else row['avg_bbox'],
                        last_appear = row['last_appear']) for index, row in df.iterrows()]
        pickle.dump(all_ctpes, pkl_file)

    print("\n" + "="*50)
    print("Pickle Index Generation Complete!")
    print(f"Source CSV: {bbox_path}")
    print(f"Output PKL: {pkl_path}")
    print(f"Total objects serialized: {len(all_ctpes)}")
    print("="*50 + "\n")