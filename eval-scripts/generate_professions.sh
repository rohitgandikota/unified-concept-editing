python eval-scripts/generate-images.py --model_name 'unbiased-scientist-attributes-male_female.pt' --num_samples 10 --device 'cuda:1' --prompts_path '/disk/u/rohit/erase-closed/data/profession1000_prompts.csv' --save_path '/disk/u/rohit/www/bias/' --from_case 0 --till_case 1000 & 

python eval-scripts/generate-images.py --model_name 'unbiased-teacher-attributes-male_female.pt' --num_samples 10 --device 'cuda:2' --prompts_path '/disk/u/rohit/erase-closed/data/profession1000_prompts.csv' --save_path '/disk/u/rohit/www/bias/' --from_case 0 --till_case 1000 & 

python eval-scripts/generate-images.py --model_name 'unbiased-librarian-attributes-male_female.pt' --num_samples 10 --device 'cuda:3' --prompts_path '/disk/u/rohit/erase-closed/data/profession1000_prompts.csv' --save_path '/disk/u/rohit/www/bias/' --from_case 0 --till_case 1000 & 

python eval-scripts/generate-images.py --model_name 'unbiased-doctor-attributes-male_female.pt' --num_samples 10 --device 'cuda:4' --prompts_path '/disk/u/rohit/erase-closed/data/profession1000_prompts.csv' --save_path '/disk/u/rohit/www/bias/' --from_case 0 --till_case 1000 & 

python eval-scripts/generate-images.py --model_name 'unbiased-comedian-attributes-male_female.pt' --num_samples 10 --device 'cuda:5' --prompts_path '/disk/u/rohit/erase-closed/data/profession1000_prompts.csv' --save_path '/disk/u/rohit/www/bias/' --from_case 0 --till_case 1000 & 

python eval-scripts/generate-images.py --model_name 'unbiased-plumber-attributes-male_female.pt' --num_samples 10 --device 'cuda:6' --prompts_path '/disk/u/rohit/erase-closed/data/profession1000_prompts.csv' --save_path '/disk/u/rohit/www/bias/' --from_case 0 --till_case 1000 & 

python eval-scripts/generate-images.py --model_name 'unbiased-engineer-attributes-male_female.pt' --num_samples 10 --device 'cuda:7' --prompts_path '/disk/u/rohit/erase-closed/data/profession1000_prompts.csv' --save_path '/disk/u/rohit/www/bias/' --from_case 0 --till_case 1000 & 
