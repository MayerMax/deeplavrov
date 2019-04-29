# deeplavrov

This is super simple framework originally developed to perform machine translation directly from python code and tune all the necessary things you may want to tune in the **seq2seq** architecture.

Project's name, **deeplavrov**, comes from Russia's Foreign Minister Sergey Lavrov who speaks several languages perfectly. :laughing:


![Image of Yaktocat](https://images.aif.ru/016/357/91bfc0ebd18e0219b430cef750454b04.jpg)

### Usage Example

Organize your datasets in such a way that all texts in the source language are in one file and texts in the target language are in another. Then create a Translator object and customize its parameters: 

```python
translator = AttentionRNNTranslator(batch_size=16, num_epochs=13)
```
Now it's ready to be fitted:

```python
translator.fit_from_file(input_file='input_train_eng.txt', target_file='target_train_ru.txt', input_val=None, target_val=None)
```

After the model has converged you can save it and restore later when you need it:

```python
translator.save('model.pickle')
translator = AttentionRNNTranslator.load('model.pickle')
```

Then you can use the model for translation:
```python
translator.translate('I was watching television when the telephone rang.')
# ['я', 'смотрела', 'телевизор', ',', 'когда', 'зазвонил', 'телефон', '.']
```

### What is done so far in the project

This is an __early stage__ of the project and not so many things are implemented:

* Word-level machine translation for datasets of arbitrary size.
* Abilility to set most of the parameters you need in a seq2seq architecture, check it when creating an object.
* Simple greedy translation.
* Attention is built-in and can not be changed manually in the current version.

### Many things to do next

We are interested in contributing to this project and are planning to implement a lot of stuff:

* Char-level support to deal with OOV
* Beam-search to make translation quality better
* More flexibility in parameters tuning
