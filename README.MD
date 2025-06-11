# Oracle Vector - RAG Demo

It demonstrates how you can perform RAG on Oracle AI Vector Search using Lanchain.

The program prompts what is the most powerful consumer grade GPU from NVIDIA in 2025. Using a model with no updated info returns previous generations (NO RAG).

After feeding the vector database using recent Wikipedia links with the latest versions, using RAG the program answers correctly (as of today, RTX 5090).

---

## 🚀 Getting Started

You can quickly get an image of Oracle Database 23ai Free (using Podman):

- [Oracle Container Registry](https://container-registry.oracle.com)  Go to **Database > free (23ai)**

```sh
podman run --name oracle32aifree -p 1521:1521 -e ORACLE_PWD=password container-registry.oracle.com/database/free:latest
```

You can use the LLM model of your choice.

In this demo, I am using Mistral-7B-Instruct-v0.3.Q4_K_M from Podman AI Lab (on my local PC) 

---

## 📋 Steps

Just run the main.py file
```sh
python main.py
```
