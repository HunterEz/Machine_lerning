import os
import urllib.request
from llama_cpp import Llama


def download_file(file_link, filename):
    # Checks if the file already exists before downloading
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(file_link, filename)
        print("File downloaded successfully.")
    else:
        print("File already exists.")

# Dowloading GGML model from HuggingFace
ggml_model_path = "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_0.gguf"
filename = "zephyr-7b-beta.Q4_0.gguf"

download_file(ggml_model_path, filename)

llm = Llama(model_path="zephyr-7b-beta.Q4_0.gguf", n_ctx=2048, n_batch=1024)

def generate_text(
    prompt="Who is the CEO of Apple?",
    max_tokens=256,
    temperature=0.1,
    top_p=0.5,
    echo=False,
    stop=["#"],
):
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        echo=echo,
        stop=stop,
    )
    output_text = output["choices"][0]["text"].strip()
    return output_text

def generate_prompt_from_template(input):
    chat_prompt_template = f"""<|im_start|>system
You are a helpful chatbot.<|im_end|>
<|im_start|>user
{input}<|im_end|>"""
    return chat_prompt_template

def make_answer(text):
    prompt = generate_prompt_from_template(text)
    result = generate_text(prompt, max_tokens = 5120)
    print(result)


query = '''
Сделай пересказ текста для ребенка на английском языке
The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin. The ultimate reason for this is Moore's law, or rather its generalization of continued exponentially falling cost per unit of computation. Most AI research has been conducted as if the computation available to the agent were constant (in which case leveraging human knowledge would be one of the only ways to improve performance) but, over a slightly longer time than a typical research project, massively more computation inevitably becomes available. Seeking an improvement that makes a difference in the shorter term, researchers seek to leverage their human knowledge of the domain, but the only thing that matters in the long run is the leveraging of computation. These two need not run counter to each other, but in practice they tend to. Time spent on one is time not spent on the other. There are psychological commitments to investment in one approach or the other. And the human-knowledge approach tends to complicate methods in ways that make them less suited to taking advantage of general methods leveraging computation.  There were many examples of AI researchers' belated learning of this bitter lesson, and it is instructive to review some of the most prominent.
In computer chess, the methods that defeated the world champion, Kasparov, in 1997, were based on massive, deep search. At the time, this was looked upon with dismay by the majority of computer-chess researchers who had pursued methods that leveraged human understanding of the special structure of chess. When a simpler, search-based approach with special hardware and software proved vastly more effective, these human-knowledge-based chess researchers were not good losers. They said that ``brute force" search may have won this time, but it was not a general strategy, and anyway it was not how people played chess. These researchers wanted methods based on human input to win and were disappointed when they did not.
A similar pattern of research progress was seen in computer Go, only delayed by a further 20 years. Enormous initial efforts went into avoiding search by taking advantage of human knowledge, or of the special features of the game, but all those efforts proved irrelevant, or worse, once search was applied effectively at scale. Also important was the use of learning by self play to learn a value function (as it was in many other games and even in chess, although learning did not play a big role in the 1997 program that first beat a world champion). Learning by self play, and learning in general, is like search in that it enables massive computation to be brought to bear. Search and learning are the two most important classes of techniques for utilizing massive amounts of computation in AI research. In computer Go, as in computer chess, researchers' initial effort was directed towards utilizing human understanding (so that less search was needed) and only much later was much greater success had by embracing search and learning.
In speech recognition, there was an early competition, sponsored by DARPA, in the 1970s. Entrants included a host of special methods that took advantage of human knowledge---knowledge of words, of phonemes, of the human vocal tract, etc. On the other side were newer methods that were more statistical in nature and did much more computation, based on hidden Markov models (HMMs). Again, the statistical methods won out over the human-knowledge-based methods. This led to a major change in all of natural language processing, gradually over decades, where statistics and computation came to dominate the field. The recent rise of deep learning in speech recognition is the most recent step in this consistent direction. Deep learning methods rely even less on human knowledge, and use even more computation, together with learning on huge training sets, to produce dramatically better speech recognition systems. As in the games, researchers always tried to make systems that worked the way the researchers thought their own minds worked---they tried to put that knowledge in their systems---but it proved ultimately counterproductive, and a colossal waste of researcher's time, when, through Moore's law, massive computation became available and a means was found to put it to good use.
In computer vision, there has been a similar pattern. Early methods conceived of vision as searching for edges, or generalized cylinders, or in terms of SIFT features. But today all this is discarded. Modern deep-learning neural networks use only the notions of convolution and certain kinds of invariances, and perform much better.
This is a big lesson. As a field, we still have not thoroughly learned it, as we are continuing to make the same kind of mistakes. To see this, and to effectively resist it, we have to understand the appeal of these mistakes. We have to learn the bitter lesson that building in how we think we think does not work in the long run. The bitter lesson is based on the historical observations that 1) AI researchers have often tried to build knowledge into their agents, 2) this always helps in the short term, and is personally satisfying to the researcher, but 3) in the long run it plateaus and even inhibits further progress, and 4) breakthrough progress eventually arrives by an opposing approach based on scaling computation by search and learning. The eventual success is tinged with bitterness, and often incompletely digested, because it is success over a favored, human-centric approach.
One thing that should be learned from the bitter lesson is the great power of general purpose methods, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are search and learning.
The second general point to be learned from the bitter lesson is that the actual contents of minds are tremendously, irredeemably complex; we should stop trying to find simple ways to think about the contents of minds, such as simple ways to think about space, objects, multiple agents, or symmetries. All these are part of the arbitrary, intrinsically-complex, outside world. They are not what should be built in, as their complexity is endless; instead we should build in only the meta-methods that can find and capture this arbitrary complexity. Essential to these methods is that they can find good approximations, but the search for them should be by our methods, not by us. We want AI agents that can discover like we can, not which contain what we have discovered. Building in our discoveries only makes it harder to see how the discovering process can be done.
'''
make_answer(query)

# For children, here's a simplified version:
# In AI research, using more computation is usually better than relying on human knowledge. 
# This is because the amount of computation available keeps getting cheaper over time. 
# Researchers often try to use human knowledge at first, but in the long run, using more computation is much more effective. 
# This can be seen in games like chess and Go, where methods that rely on lots of search and learning have beaten world champions. 
# In speech recognition and computer vision, early methods tried to mimic how humans see or hear things, 
# but now deep learning neural networks are used instead because they use less human knowledge and perform better. 
# This is a lesson we still need to learn in AI research, because sometimes researchers try to build in what they think they know, 
# but this can hold back progress in the long run. Instead, we should focus on using methods that can find and capture complexity, 
# rather than trying to understand everything about the world. These methods can find good approximations, 
# but the search for them should be done by our methods, not by us. We want AI agents that can discover things like humans do, 
# not ones that already know what we know.

query = '''
TRANSLATE TO RUSSIAN: 
The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin. The ultimate reason for this is Moore's law, or rather its generalization of continued exponentially falling cost per unit of computation. Most AI research has been conducted as if the computation available to the agent were constant (in which case leveraging human knowledge would be one of the only ways to improve performance) but, over a slightly longer time than a typical research project, massively more computation inevitably becomes available. Seeking an improvement that makes a difference in the shorter term, researchers seek to leverage their human knowledge of the domain, but the only thing that matters in the long run is the leveraging of computation. These two need not run counter to each other, but in practice they tend to. Time spent on one is time not spent on the other. There are psychological commitments to investment in one approach or the other. And the human-knowledge approach tends to complicate methods in ways that make them less suited to taking advantage of general methods leveraging computation.  There were many examples of AI researchers' belated learning of this bitter lesson, and it is instructive to review some of the most prominent.
In computer chess, the methods that defeated the world champion, Kasparov, in 1997, were based on massive, deep search. At the time, this was looked upon with dismay by the majority of computer-chess researchers who had pursued methods that leveraged human understanding of the special structure of chess. When a simpler, search-based approach with special hardware and software proved vastly more effective, these human-knowledge-based chess researchers were not good losers. They said that ``brute force" search may have won this time, but it was not a general strategy, and anyway it was not how people played chess. These researchers wanted methods based on human input to win and were disappointed when they did not.
A similar pattern of research progress was seen in computer Go, only delayed by a further 20 years. Enormous initial efforts went into avoiding search by taking advantage of human knowledge, or of the special features of the game, but all those efforts proved irrelevant, or worse, once search was applied effectively at scale. Also important was the use of learning by self play to learn a value function (as it was in many other games and even in chess, although learning did not play a big role in the 1997 program that first beat a world champion). Learning by self play, and learning in general, is like search in that it enables massive computation to be brought to bear. Search and learning are the two most important classes of techniques for utilizing massive amounts of computation in AI research. In computer Go, as in computer chess, researchers' initial effort was directed towards utilizing human understanding (so that less search was needed) and only much later was much greater success had by embracing search and learning.
In speech recognition, there was an early competition, sponsored by DARPA, in the 1970s. Entrants included a host of special methods that took advantage of human knowledge---knowledge of words, of phonemes, of the human vocal tract, etc. On the other side were newer methods that were more statistical in nature and did much more computation, based on hidden Markov models (HMMs). Again, the statistical methods won out over the human-knowledge-based methods. This led to a major change in all of natural language processing, gradually over decades, where statistics and computation came to dominate the field. The recent rise of deep learning in speech recognition is the most recent step in this consistent direction. Deep learning methods rely even less on human knowledge, and use even more computation, together with learning on huge training sets, to produce dramatically better speech recognition systems. As in the games, researchers always tried to make systems that worked the way the researchers thought their own minds worked---they tried to put that knowledge in their systems---but it proved ultimately counterproductive, and a colossal waste of researcher's time, when, through Moore's law, massive computation became available and a means was found to put it to good use.
In computer vision, there has been a similar pattern. Early methods conceived of vision as searching for edges, or generalized cylinders, or in terms of SIFT features. But today all this is discarded. Modern deep-learning neural networks use only the notions of convolution and certain kinds of invariances, and perform much better.
This is a big lesson. As a field, we still have not thoroughly learned it, as we are continuing to make the same kind of mistakes. To see this, and to effectively resist it, we have to understand the appeal of these mistakes. We have to learn the bitter lesson that building in how we think we think does not work in the long run. The bitter lesson is based on the historical observations that 1) AI researchers have often tried to build knowledge into their agents, 2) this always helps in the short term, and is personally satisfying to the researcher, but 3) in the long run it plateaus and even inhibits further progress, and 4) breakthrough progress eventually arrives by an opposing approach based on scaling computation by search and learning. The eventual success is tinged with bitterness, and often incompletely digested, because it is success over a favored, human-centric approach.
One thing that should be learned from the bitter lesson is the great power of general purpose methods, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are search and learning.
The second general point to be learned from the bitter lesson is that the actual contents of minds are tremendously, irredeemably complex; we should stop trying to find simple ways to think about the contents of minds, such as simple ways to think about space, objects, multiple agents, or symmetries. All these are part of the arbitrary, intrinsically-complex, outside world. They are not what should be built in, as their complexity is endless; instead we should build in only the meta-methods that can find and capture this arbitrary complexity. Essential to these methods is that they can find good approximations, but the search for them should be by our methods, not by us. We want AI agents that can discover like we can, not which contain what we have discovered. Building in our discoveries only makes it harder to see how the discovering process can be done.
'''
make_answer(query)

# 70 лет исследований искуственного интеллекта преподают нам самую важную лекцию - методы, которые используют вычисления, в конечном счете оказываются наиболее эффективными и этого достаточно. Эта причина заключается в том, что согласно теореме Мурза, или ее продолжению - падению стоимости вычислений на единицу ресурса, больше вычислений становится доступным. Большинство исследований искуственного интеллекта велось как будто количество вычислений, доступных агенту, постоянно. Однако, в более длительной перспективе, огромное количество вычислений становится доступным. Исследователи ищут методы, которые используют наше знание о домене, но только в короткосрочной перспективе. В долгосрочной перспективе, однако, только использование вычислений имеет значение. Эти две области не должны конфликтовать друг с другом, но часто это происходит в практике. Время, которое уделяется одной области, вре 
# занимаемое другим. Существуют психологические предубеждения в пользу одного подхода или другого. И методы, основанные на знании человека, часто осложняют методы, использующие вычисления. Впрочем, эти две области не должны конфликтовать друг с другом, но в практике часто это происходит.
# В шахматах, методы, которые победили чемпиона мира по шахматам Каспарова в 1997 году, были основаны на массовом и глубоком поиске. Эти методы были восприняты критикой большинства исследователей компьютерных шахмат, которые преследовали методы, основанные на знании человека о структуре шахматной доски. Когда простой и эффективный подход с использованием специальной аппаратуры и программного обеспечения оказался гораздо более эффективным, исследователи компьютерных шахмат, основанные на знании человека, не были хорошими проигравшими. Они считали, что "сыла" поиска может выиграть сейчас, но это не был общий метод, и в любом случае он не соответствует способу игры людей. Исследователи компьютерных шахмат, основанные на знании человека, хотели, чтобы методы, основанные на знании человека, победили, и были расстроены, когда это не произошло.
# Аналогичный паттерн исследований был замечен в компьютерном го, только с задержкой в 20 лет. Большинство первоначальных усилий были направлены на избежание поиска и на использование знания о домене или специфических особенностях игры. Однако все эти усилия оказались бесполезными, как только поиск был применён эффективно на больших масштабах. Также важной была роль обучения по самообучению, которая играла важную роль в других играх и даже в шахматах, хотя она не сыграла крупной роли в программе 1997 года, первой победившей чемпиона мира по шахматам.

# пересказ статьи "КОМПЬЮТЕРНОЕ МОДЕЛИРОВАНИЕ ПРОЦЕССА КОРМЛЕНИЯ СОБАК В ЗАКРЫТЫХ ВОЛЬЕРАХ В УСЛОВИЯХ ПАНДЕМИИ"

query = ''' Сделай пересказ статьи:
В условиях пандемии ведомственные кинологические службы испытывают значительные затруднения при кормлении служебных собак. В статье приведены результаты изучения существующих систем кормления собак и описана разработка автоматизированной системы с использованием программируемого логического контроллера (ПЛК) Omron, удовлетворяющей требованиям нормативных документов кинологических служб. Приведена разработанная технологическая схема с подбором и маркировкой оборудования. Для упрощения монтажа оборудование
разбито на модули. Приведена таблица адресов сигналов, используемых в промышленных контроллерах серии Omron. Приведена система логических уравнений для управления основными модулями привода устройств оборудования. Цель исследования: разработка проекта автоматизированной системы кормления собак в питомниках ФСИН России на базе программируемых логических контроллеров и сокращения использования человеческих ресурсов. Материалы
и методы: методика кормления служебных собак согласована с Приказами ФСИН России
№ 330 от 13 мая 2008 г., № 570 от 4 июля 2018 г. Использовались теория конечных автоматов, теория синтеза логических уравнений и методика построения лестничных диаграмм. Программное обеспечение разработано с помощью CX-One. В качестве корма предлагается использовать сухой гранулированный корм для собак. Результаты: разработан проект автоматизированной системы на 5 собак, включающий подбор технологического оборудования. Система протестирована в режиме симуляции программного обеспечения. Практическая значимость: внедрение данной системы в ведомственных организациях позволит сократить затраты времени на дозирование корма, уменьшить вероятность ошибки, связанной с человеческим фактором, а также
в условиях пандемии обеспечить процесс кормления служебных собак при снижении трудозатрат обслуживающего персонала.)
'''
make_answer(query)

# Пересказ ЛЛМ обрывается в определенный момент. Оценка пересказов: мой 8, ЛЛМ 1