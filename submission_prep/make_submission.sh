source config.txt
cur_folder=$(pwd)

# Copy files
cp "$source_model" ../kaggle_submissions/model.zip
cp "$source_agent" ../kaggle_submissions/agent_policy.py
cp -r "$rewards_location" ../kaggle_submissions

#Check dependencies
cd ../kaggle_submissions
sudo rm -r luxai2021
sudo rm -r stable_baselines3
python3 download_dependencies.py

#Make submission file
cd ..
tar -czf "$target_name" -C kaggle_submissions .
cd "$cur_folder"
