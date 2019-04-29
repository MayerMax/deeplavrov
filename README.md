# deeplavrov

This is super simple framework originally developed to complete machine translation directly from python code and tune all the necessary things you may want to tune in the **seq2seq** architecture.

**deeplavrov** naming comes from a Russian's Federation Foreign Secretary Sergey Lavrov who speaks perfect several languages. :laughing:


![Image of Yaktocat](https://images.aif.ru/016/357/91bfc0ebd18e0219b430cef750454b04.jpg)

### Usage Example

Organize your datasets in such a way that all texts for source language in one file and target language in another. Then create an object and customize it's parameters: 

```python
translator = AttentionRNNTranslator(batch_size=16, num_epochs=13)
```
Now it is ready to be fitted:

```python
translator.fit_from_file(input_file='input_train_eng.txt', target_file='target_train_ru.txt', input_val=None, target_val=None)
```

After the model converged you can save it and restore when you need:

```python
translator.save('model.pickle')
translator = AttentionRNNTranslator.load('model.pickle')
```

After all, perform the translation and use where you need:
```python
translator.translate('I was watching television when the telephone rang.')
# ['я', 'смотрела', 'телевизор', ',', 'когда', 'зазвонил', 'телефон', '.']
```

### What is done so far in the project

This is an __early stage__ of the project and not so many things are implemented:

* Word-level machine translation for datasets of arbitrary size.
* Able to set most of the parameters you need in a seq2seq architecture, check it when create an object.
* Simple greedy translation.
* Attention is built-in and can not be changed manually in the current version.

### Many things to do next

We are interested in contributing to this project and are planning to implement many stuff:

* Char-level support to deal with OOV
* Beam-search to make translation quality better
* More flexibility in tuning parameters
