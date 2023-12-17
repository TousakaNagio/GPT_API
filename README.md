# GPT_API
A custom API for OpenAI GPT.

## Install

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/TousakaNagio/GPT_API.git
cd GPT_API
```

2. Install Package
```Shell
pip install openai
```

3. Install additional packages for video processing.
```
pip install tqdm
pip install numpy
```

## Usage

1. Follow the sample code in main.py to chat with gpt api.

2. We provide gpt_retrieve.py as an example for clip-wise retrieval with gpt-4v.

3. If you call reset, you can start a new conversation with GPT, but the previous conversation will be lost. You also need to re-configure the GPT system and schema.
