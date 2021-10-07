source config.txt
cur_folder=$(pwd)

# Copy files
cp "$source_model" ../kaggle_submissions/model.zip
cp "$source_agent" ../kaggle_submissions

#Check dependencies
cd ../kaggle_submissions
python download_dependencies.py

#Make submission file
cd ..
tar -czf "$target_name" -C kaggle_submissions .
cd "$cur_folder"