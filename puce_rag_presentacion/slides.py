"""
Created by Analitika at 07/09/2024
contact@analitika.fr
"""
import os

from dotenv import load_dotenv

load_dotenv()
aws_path = os.getenv("AWS_PATH")

favicon = f"{aws_path}/DS_Imagen_de_marca/logos/favicondsazul.png"
sample_markdown = fr"""
<section style="background-color: #0772CA; color: white; text-align: center;">
    <div style="margin-top: 15vh;">
        <h1 style="font-size: 1.5em; font-weight: bold;">Fundamentos de LLMs</h1>
        <h1 style="font-size: 1.5em; font-weight: bold;">Comparación Semántica y RAG</h1>
        <h2 style="font-size: 1.0em; font-style: italic;">Escuela de Verano - ML</h2>
        <h3 style="font-size: 1.2em;">Pontificia Universidad Católica - Quito</h3>
        <p style="font-size: 1.1em; margin-top: 1.5em;">13 Septiembre 2024</p>
    </div>
</section>
---
<!-- .slide: data-background-image="{aws_path}/DS_Imagen_de_marca/SLIDES/Presentacion_V1_01.png" data-background-size="115% 100%" data-background-position="center" -->

<section style="position: relative; top: 20%; left: 10%; display: flex; justify-content: space-between; align-items: center; width: 90%;">
    <!-- Left content: Text -->
    <div style="position: absolute; top: 30%; left: 10%; text-align: left; color: white; max-width: 50%;">
        <h2 style="color: white;">¿Quién soy?</h2>
        <strong>Eduardo Cepeda, Ph.D.</strong> <br>
        <em>CEO & Founder</em> <br>
        📞 +33 (0)6 50 90 01 49 <br>
        ✉️ <a href="mailto:eduardo@datoscout.ec" style="color: white;">eduardo@datoscout.ec</a> <br>
        🌐 <a target="_blank" href="http://www.datoscout.ec" style="color: white;">www.datoscout.ec</a>
    </div>
    <!-- Image positioned on the right side using CSS -->
    <div style="position: absolute; top: 20%; right: 10%; width: 30%;">
        <img src="{aws_path}/PRESENTACION+PUCE/yo.png" alt="Eduardo Cepeda" style="width: 100%; height: auto; border-radius: 8px;">
    </div>    
</section>
--
## Trayectoria

<img src="{aws_path}/PRESENTACION+PUCE/Presentation1.png" alt="trayectoria" style="width: 1500px; height: auto;">

---
## Plan de la charla 
- Hablar de LLMs, su evolución y progreso  <!-- .element: class="fragment" data-fragment-index="0" -->
- Ganar intuición sobre RAG <!-- .element: class="fragment" data-fragment-index="0" -->
- Prácticar con ejemplos <!-- .element: class="fragment" data-fragment-index="1" -->
- Responder preguntas <!-- .element: class="fragment" data-fragment-index="2" -->

### Objetivo
- Crear curiosidad y dar elementos DIY

---
<!-- Qué es RoPE self-extended -->
<!-- 4-8K tokens ~ 12 pages -> 1M ~ 1k pages -->
### Motivación: Mejoras en capacidad 

<img src="{aws_path}/PRESENTACION+PUCE/09-context-window.png" alt="Context Windows" style="width: 900px; height: auto;">

- La mayor parte de la información del mundo es privada <br> <!-- .element: class="fragment" data-fragment-index="0" -->
- ... pero la podemos "inyectar" a un LLM <!-- .element: class="fragment" data-fragment-index="1" -->
--
### Nuevo paradigma: 
LLM como Sistema Operativo

<img src="{aws_path}/PRESENTACION+PUCE/01-RAG-as-a-OS.png" alt="LLMs as OS" style="width: 900px; height: auto;"><!-- .element: class="fragment" data-fragment-index="0" -->
- RAG puede reemplazar/completar el fine-tunning <!-- .element: class="fragment" data-fragment-index="1" -->
---
**R**etrieval **A**ugmented **G**eneration

<img src="{aws_path}/PRESENTACION+PUCE/02-Schema.png" alt="schema" style="width: 900px; height: auto;">
---
## Perspectiva

| Nivel Básico                        | Avanzado                               | 
|-------------------------------------|----------------------------------------|
|                                     | - Transformación de preguntas          |
|                                     | - Ruteo                                |
|                                     | - Construcción de preguntas            |
| - Indexación                        | - Indexación                           |
| - Recuperación                      | - Recuperación                         |
| - Generación                        | - Generación                           |

---
<img src="{aws_path}/PRESENTACION+PUCE/03-Document-Loading.png" alt="LLMs as OS" style="width: 900px; height: auto;">
--
<img src="{aws_path}/PRESENTACION+PUCE/04-Numerical-representation.png" alt="numerical representation" style="width: 900px; height: auto;">
--
<img src="{aws_path}/PRESENTACION+PUCE/05-Loading-splitting-embedding.png" alt="load-splitting" style="width: 900px; height: auto;">
--
<img src="{aws_path}/PRESENTACION+PUCE/07-Vectorestore.png" alt="vectorstore" style="width: 900px; height: auto;">
---
### Manos a la obra:
- Modelo `distilbert-base-multilingual-cased` 
  - Fuente (checkpoints): <a target="_blank" href="https://huggingface.co/distilbert/distilbert-base-multilingual-cased">HuggingFace 🤗</a>
  - Información:  <a target="_blank" href="https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation">GitHub <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub Logo" style="width: 50px; height: auto;"></a>

- DistilBERT es un modelo multilingüe que soporta 104 idiomas.
- Tiene 6 capas, de dimensión 768 y 12 cabezas de atención.
- 134 millones de parámetros (vs 177 millones de mBERT-base).
- DistilBERT es el doble de rápido que mBERT-base.
--
### Aparte técnico
<img src="{aws_path}/PRESENTACION+PUCE/Prod-optim.png" alt="Production Optimisation" style="width: 700px; height: auto;">
<p style="font-size: 0.8em; text-align: center;">
  Para espíritus curiosos: <a target="_blank" href="https://medium.com/@quocnle/how-we-scaled-bert-to-serve-1-billion-daily-requests-on-cpus-d99be090db26">Medium</a>
</p>
<section>
  <h3 style="color: #007BFF; font-size: 1.0em; text-transform: none;">Optimización en producción</h3>
  <ul style="font-size: 0.9em;">
    <li><strong>Distilación:</strong> Entrenamiento supervisado de un modelo más pequeño.</li>
    <li><strong>Quantización:</strong> Reducir la precisión de los pesos - reducción de memoria.</li>
    <li><strong>Pruning:</strong> Eliminar conexiones o pesos irrelevantes.</li>
  </ul>
</section>
---
<h3 style="color: #007BFF; font-size: 1.0em; text-transform: none;">Estructura de un modelo LLMs</h3>
<img src="{aws_path}/PRESENTACION+PUCE/08-trainable-parameters.png" alt="LLM structure" style="width: 800px; height: auto;">

  <h3 style="font-size: 0.9em; margin-left: 50px; text-align: left; color: #007BFF; text-transform: none;">Recursos Importantes</h3>

  <p style="font-size: 0.7em; margin-left: 50px; text-align: left;">Artículo original: 
    <a target="_blank" href="https://arxiv.org/abs/1706.03762" style="font-size: 0.7em; text-align: left; color: #FF5733; text-decoration: none;">
      Vaswani, A. "Attention is all you need." Advances in Neural Information Processing Systems (2017)
    </a>
  </p>

  <div style="font-size: 0.9em; text-align: left; list-style-type: disc; margin-left: 50px;">
      <ul style="font-size: 0.9em; text-align: left; list-style-type: disc; margin-left: 5px;">
        <li>
          <a target="_blank" href="https://nlp.seas.harvard.edu/annotated-transformer/" style="font-size: 0.9em; text-align: left; color: #3498DB; text-decoration: none;">
            Transformers Anotados
          </a>
        </li>
        <li>
          <a target="_blank" href="https://www.oreilly.com/library/view/natural-language-processing/9781098136789/" style="font-size: 0.9em; text-align: left; color: #3498DB; text-decoration: none;">
            Natural Language Processing 🤗
          </a>
        </li>
        <li>
          <a target="_blank" href="https://www.packtpub.com/en-us/product/transformers-for-natural-language-processing-9781803247335" style="font-size: 0.9em; text-align: left; color: #3498DB; text-decoration: none;">
            Transformers for NLP
          </a>
        </li>
      </ul>
  </div>
--

<h3 style="color: #007BFF; font-size: 1.0em; text-transform: none;">Completar la frase [MASK]</h3>
<pre><code class="language-python" data-line-numbers="1-2|3-5|6-9|10-12">import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
model_str = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_str)
model = AutoModelForMaskedLM.from_pretrained(model_str)
text = "Machine Learning es el mejor [MASK]"
encoded_input = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    model_output = model(**encoded_input)
masked_index = torch.where(encoded_input["input_ids"] == tokenizer.mask_token_id)[1]
logits = model_output.logits[0, masked_index, :]
top_5_token_ids = logits.topk(5, dim=-1).indices[0].tolist()
</code></pre>

--
<div>
  <h3 style="font-size: 1.1em; color: #007BFF; text-transform: none;">Texto Original:</h3>
  <p style="font-size: 1.0em; font-weight: bold; color: #FF5733;">Machine Learning es el mejor [MASK]</p>
  <h3 style="font-size: 1.1em; color: #007BFF; text-transform: none;">Top 5 Predicciones</h3>
  <ul style="font-size: 1.0em;">
    <li>Machine Learning es el mejor <span style="color: #3498DB;">trabajo</span></li>
    <li>Machine Learning es el mejor <span style="color: #E74C3C;">desarrollo</span></li>
    <li>Machine Learning es el mejor <span style="color: #2ECC71;">método</span></li>
    <li>Machine Learning es el mejor <span style="color: #F39C12;">proyecto</span></li>
    <li>Machine Learning es el mejor <span style="color: #9B59B6;">nivel</span></li>
  </ul>
</div>
---
<h3 style="color: #007BFF; font-size: 1.0em; text-transform: none;">Generadores de Features</h3>

<img src="{aws_path}/PRESENTACION+PUCE/06-transformers-as-feature-extractors.png" alt="LLM Features" style="width: 800px; height: auto;">

<p style="font-size: 0.9em; text-align: left; margin-left: 50px;">Artículo original: 
<a target="_blank" href="https://arxiv.org/abs/1706.03762" style="font-size: 0.7em; text-align: left; color: #FF5733; text-decoration: none;">
  Vaswani, A. "Attention is all you need." Advances in Neural Information Processing Systems (2017)
</a>
</p>
--
<h3 style="color: #007BFF; font-size: 1.0em; text-transform: none;">Calcular embeddings</h3>
<pre><code class="language-python" data-line-numbers="1-3|4-7|8">import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
encoded_input = tokenizer(text_, return_tensors="pt")
with torch.no_grad():
    model_output = model(**encoded_input)
embedding_1 = model_output.last_hidden_state.mean(dim=1).squeeze().numpy()
similarity = cosine_similarity([text_reference_embeddings], [embedding_1])[0][0]
</code></pre>
--
<div>
    <h3 style="font-size: 0.8em; color: #007BFF;">Data Science</h3>
    <ul style="font-size: 0.6em;">
      <li>El análisis de datos permite descubrir tendencias ocultas en grandes conjuntos de información.</li>
      <li>La inteligencia artificial y el aprendizaje automático son pilares fundamentales en la ciencia de datos.</li>
      <li>La visualización de datos ayuda a comprender patrones complejos de manera intuitiva.</li>
    </ul>
</div>
<div>
    <h3 style="font-size: 0.8em; color: #FF5733;">Viajes & Aventura</h3>
    <ul style="font-size: 0.6em;">
      <li>Las montañas del Himalaya ofrecen rutas de senderismo desafiantes y paisajes impresionantes.</li>
      <li>La exploración de la selva amazónica revela una biodiversidad asombrosa y culturas indígenas únicas.</li>
      <li>Un crucero por el Mediterráneo permite descubrir antiguas civilizaciones y playas soleadas.</li>
    </ul>
</div>
    <h3 style="font-size: 0.8em; color: #8E44AD;">Literatura</h3>
    <ul style="font-size: 0.6em;">
      <li>Gabriel García Márquez es un ícono de la literatura.</li>
      <li>El realismo mágico es un estilo literario fascinante.</li>
      <li>La poesía expresa emociones profundas.</li>
    </ul>
---
## Resultados

<p style="font-size: 0.7em; text-align: left;">Disciplina científica centrada en el análisis de grandes fuentes de datos para extraer información, comprender la realidad y descubrir patrones para tomar decisiones.</p>

$\scriptsize \rm{{cosine\\,similarity}} = \frac{{ Emb_1 \cdot Emb_2 }} {{ ||Emb_1|| \\, ||Emb_2|| }}$

<img src="{aws_path}/PRESENTACION+PUCE/similaridad_sematica.png" alt="LLM Features" style="width: 600px; height: auto;">



--
<h3 style="color: #007BFF; font-size: 1.0em; text-transform: none;">Generadores de Features</h3>

<img src="{aws_path}/PRESENTACION+PUCE/embeddings_projection.png" alt="projections" style="width: 900px; height: auto;">
---
**R**etrieval **A**ugmented **G**eneration

<img src="{aws_path}/PRESENTACION+PUCE/02-Schema.png" alt="schema" style="width: 900px; height: auto;">
---
<!-- .slide: data-background-image="{aws_path}/PRESENTACION+PUCE/RAG-FULL.png" -->
---

<h3>Retroalimentación</h3>
<img src="{aws_path}/PRESENTACION+PUCE/flujo_final.png" alt="flujo_final" style="width: 9000px; height: auto;">

<img src="{aws_path}/PRESENTACION+PUCE/Fillout QR Code.png" alt="projections" style="width: 150px; height: auto;">
<a target="_blank" href="https://forms.fillout.com/t/tmNM7SUWuJus" style="font-size: 0.7em; text-align: left; color: #0772CA; text-decoration: none;">
  https://forms.fillout.com/t/tmNM7SUWuJus
</a>

---
## The end
<img src="{aws_path}/PRESENTACION+PUCE/cari.png" alt="projections" style="width: 700px; height: auto;">
<div style="font-size: 0.9em; text-align: left; list-style-type: disc; margin-left: 200px;">
<p>Encuéntrame en <a target="_blank" href="https://www.linkedin.com/in/educep/">LinkedIn <img src="{aws_path}/PRESENTACION+PUCE/LILOGO.png" alt="GitHub Logo" style="width: 50px; height: auto;"></a> <a href="mailto:eduardo@datoscout.ec">eduardo@datoscout.ec ✉️</a></p>
<p>Código: <a target="_blank" href="https://github.com/educep/puce_rag_presentacion">GitHub /educep <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub Logo" style="width: 50px; height: auto;"></a></p>
<p>Slides: <a target="_blank" href="https://intro-rag.datoscout.ec">https://intro-rag.datoscout.ec <img src="{aws_path}/DS_Imagen_de_marca/logos/DS+logo.png" alt="GitHub Logo" style="width: 50px; height: auto;"></a></p>
</div>
"""