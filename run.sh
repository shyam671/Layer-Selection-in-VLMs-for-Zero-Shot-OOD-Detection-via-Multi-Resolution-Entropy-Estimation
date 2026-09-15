datasets=(oasis-near-ct oasis-far-heart oasis-far-chaos) 

for d in "${datasets[@]}"; do
   python main.py --text_prompt multi-mod --score MOD --dataset_dir "$d" 
done