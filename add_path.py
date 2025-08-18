import os 

# input_path = r'C:\Users\oft\Documents\ShenZhenCup\output\Q2'

def change_name_F(input_path):
    for _, dirnames, filenames in os.walk(input_path):
        for filename in filenames:
            if filename.endswith('.csv'):
                if 'Path' in filename:
                    continue
                filepath = os.path.join(input_path, filename)
                print(f"Find {filepath}")
                part1, part2 = filename.split('-')
                change_name = part1 + '-Path' + part2
                change_path = os.path.join(input_path, change_name)
                os.rename(filepath, change_path)
                print(f"Successfully change from {filepath} to {change_path}")

        for dirname in dirnames:
            change_name_F(os.path.join(input_path, dirname))

change_name_F(r'C:\Users\oft\Documents\ShenZhenCup\output\Q2')