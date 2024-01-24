python validation_spider.py --tag '350M no-compile' --checkpoint_path '/home/vendala/_Models/nsql-350M/model_int8.pth'
python validation_spider.py --tag '2B no-compile' --checkpoint_path '/home/vendala/_Models/nsql-2B/model_int8.pth'
python validation_spider.py --tag '6B no-compile' --checkpoint_path '/home/vendala/_Models/nsql-6B/model_int8.pth'
python validation_spider.py --tag '6B+350M no-compile k=5' --speculate_k=5 --checkpoint_path '/home/vendala/_Models/nsql-6B/model_int8.pth' --draft_checkpoint_path '/home/vendala/_Models/nsql-350M/model_int8.pth'
python validation_spider.py --tag '6B+350M no-compile k=20' --speculate_k=20 --checkpoint_path '/home/vendala/_Models/nsql-6B/model_int8.pth' --draft_checkpoint_path '/home/vendala/_Models/nsql-350M/model_int8.pth'

python validation_spider.py --tag '350M compile' --compile --checkpoint_path '/home/vendala/_Models/nsql-350M/model_int8.pth'
python validation_spider.py --tag '2B compile' --compile --checkpoint_path '/home/vendala/_Models/nsql-2B/model_int8.pth'
python validation_spider.py --tag '6B compile' --compile --checkpoint_path '/home/vendala/_Models/nsql-6B/model_int8.pth'
python validation_spider.py --tag '6B+350M compile k=5' --speculate_k=5 --compile --checkpoint_path '/home/vendala/_Models/nsql-6B/model_int8.pth' --draft_checkpoint_path '/home/vendala/_Models/nsql-350M/model_int8.pth'
python validation_spider.py --tag '6B+350M compile k=20' --speculate_k=20 --compile --checkpoint_path '/home/vendala/_Models/nsql-6B/model_int8.pth' --draft_checkpoint_path '/home/vendala/_Models/nsql-350M/model_int8.pth'

python validation_spider.py --tag '350M prefill' --compile --compile_prefill --checkpoint_path '/home/vendala/_Models/nsql-350M/model_int8.pth'
python validation_spider.py --tag '6B prefill' --compile --compile_prefill --checkpoint_path '/home/vendala/_Models/nsql-6B/model_int8.pth'

python validation_spider.py --tag '350M cache' --compile --prefill_cache --compile_prefill --checkpoint_path '/home/vendala/_Models/nsql-350M/model_int8.pth'
python validation_spider.py --tag '6B cache' --compile --prefill_cache --compile_prefill --checkpoint_path '/home/vendala/_Models/nsql-6B/model_int8.pth'






