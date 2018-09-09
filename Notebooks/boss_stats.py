
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class Stats():
    """
    Process godville data and show statistics.

    Shows various statistics and plots.
    """

    def __init__(self):
        """Init."""
        self.data = pd.read_excel('Godville_boss.xlsx', encoding='windows-1251')
        self.preprocess()

    def preprocess(self):
        """Preprocessing."""
        self.data.drop(['Unnamed: 8'], axis=1, inplace=True)
        self.data.loc[self.data['Тип подзема'] == 'Термодинамики (аквариум)', 'Тип подзема'] = 'Термодинамики'
        self.data.loc[self.data['Тип подзема'] == 'Спешка', 'Тип подзема'] = 'Спешки'
        self.data.loc[self.data['Тип подзема'] == '', 'Тип подзема'] = 'Спешки'
        self.data['Тип подзема'] = self.data['Тип подзема'].apply(lambda x: x.strip())

        self.data['Кусок мне'] = self.data['Кусок мне'].apply(lambda x: x.strip() if str(x) != 'nan' else x)
        self.data['Куски не мне'] = self.data['Куски не мне'].apply(lambda x: x.strip() if str(x) != 'nan' else x)
        self.data['Имя босса'] = self.data['Имя босса'].apply(lambda x: x.strip() if str(x) != 'nan' else x)

        self.data.loc[self.data['Куски не мне'] == 'лапа', 'Куски не мне'] = 'лапу'
        self.data.loc[self.data['Куски не мне'] == 'шкура', 'Куски не мне'] = 'шкуру'
        self.data.loc[self.data['Кусок мне'] == 'лапа', 'Кусок мне'] = 'лапу'
        self.data.loc[self.data['Кусок мне'] == 'шкура', 'Кусок мне'] = 'шкуру'
        self.data.loc[self.data['Имя босса'] == 'Загадки', 'Тип подзема'] = 'Загадки'
        self.data.loc[self.data['Имя босса'] == 'Загадки', 'Имя босса'] = None
        self.data['Кусок мне'] = self.data['Кусок мне'].str.replace('лапа', 'лапу')
        self.data['Кусок мне'] = self.data['Кусок мне'].str.replace('шкура', 'шкуру')
        self.data['Куски не мне'] = self.data['Куски не мне'].str.replace('лапа', 'лапу')
        self.data['Куски не мне'] = self.data['Куски не мне'].str.replace('шкура', 'шкуру')

    def plot_dungeon_counts(self):
        plt.xkcd(0.5, 100, 1);
        plt.figure(figsize=(16, 12));
        self.data['Тип подзема'].value_counts().plot('barh');
        plt.title('Количество типов подземов');

    def plot_boss_pie(self):
        no_boss_count = self.data['Имя босса'].isnull().sum()
        some_boss_count = np.sum(self.data['Имя босса'].isnull() == False)
        labels = 'Без финала', 'С финалом'
        sizes = [no_boss_count, some_boss_count]
        explode = (0, 0.1)

        plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Доля подземов с найденным финалом')

    def plot_boss_counts(self):
        plt.xkcd(0.5, 100, 1);
        plt.figure(figsize=(16, 12));
        sns.countplot(y=self.data['Имя босса'], order=self.data['Имя босса'].value_counts().index.tolist());
        plt.title('Количество финалов по названиям');
        plt.xlabel('Количество')

    def print_boss_part(self):
        """Показывает сколько уникальных боссов можно было бы собрать без учета миников и ошмётков."""
        me = []
        not_me = []
        for boss in sorted(self.data['Имя босса'].value_counts().index.tolist()):
            df = self.data.loc[self.data['Имя босса'] == boss]
            a = [i.split(',') for i in df['Кусок мне'].unique() if str(i) != 'nan']
            a = [i.strip() for j in a for i in j]
            b = [i.split(',') for i in df['Куски не мне'].unique() if str(i) != 'nan']
            b = [i.strip() for j in b for i in j]
            if len([i for i in list(set(a)) if 'мини' not in i and 'ошмёток' not in i]) == 10:
                me.append(boss)
            if len([i for i in list(set(b)) if 'мини' not in i and 'ошмёток' not in i]) == 10:
                not_me.append(boss)
        print('Я мог бы полностью собрать следующих боссов:', ', '.join(me))
        print('Мои союзники могли бы полностью собрать следующих боссов:', ', '.join(not_me))

    def plot_skill_count(self):
        data = self.data.loc[self.data['Первая способность'].isnull() == False]
        skills = list(data['Первая способность'].values) + list(data['Вторая'].values) + list(data['Третья'].values)
        df = pd.DataFrame(Counter(skills).most_common(), columns=['Способность', 'Количество'])
        df = df.loc[df['Способность'] != 'Фингалист']
        df.set_index('Способность').plot(kind='barh', color='teal', figsize=(16, 12));
        plt.title('Самые встречаемые способности');

        skills = data['Первая способность'] + ', ' + data['Вторая'] + ', ' + data['Третья']
        skills = skills.apply(lambda x: ', '.join(sorted(x.split(', '))))
        Counter(list(skills.values)).most_common()
        print('Топ тройки способностей')
        for i in Counter(list(skills.values)).most_common(3):
            print(f'Тройка способностей: {i[0]}. Количество: {i[1]}')

    def print_skill_per_dungeon(self):
        pairs = []
        for i in self.data['Тип подзема'].unique():
            df = self.data.loc[(self.data['Тип подзема'] == i) & (self.data['Первая способность'].isnull() == False)]
            skills = list(df['Первая способность'].values) + list(df['Вторая'].values) + list(df['Третья'].values)
            df1 = pd.DataFrame(Counter(skills).most_common(), columns=['Способность', 'Количество'])
            df1['Количество'] = df1['Количество'] / np.sum(df1['Количество'])
            df1 = df1.loc[df1['Количество'] > 0.12]
            if df1.shape[0] > 0:
                pairs.append((i, df1['Способность'][0], float(df1['Количество'][0]) * 100))

        for i in pairs:
            print(f'В подземе {i[0]} было {i[2]:.4f}% финалов со способностью {i[1]}.')

    def print_skill_per_boss(self):
        pairs = []
        for i in self.data['Имя босса'].unique():
            df = self.data.loc[(self.data['Имя босса'] == i) & (self.data['Первая способность'].isnull() == False)]
            skills = list(df['Первая способность'].values) + list(df['Вторая'].values) + list(df['Третья'].values)
            df1 = pd.DataFrame(Counter(skills).most_common(), columns=['Способность', 'Количество'])
            df1['Количество'] = df1['Количество'] / np.sum(df1['Количество'])
            df1 = df1.loc[df1['Количество'] > 0.12]
            if df1.shape[0] > 0:
                pairs.append((i, df1['Способность'][0], float(df1['Количество'][0]) * 100))

        for i in pairs:
            print(f'Среди финалов {i[0]} было {i[2]:.4f}% боссов со способностью {i[1]}.')

    def show_all(self):
        self.plot_dungeon_counts()
        plt.show()
        print('')
        print('*'* 50)
        print('')
        self.plot_boss_pie()
        plt.show()
        print('')
        print('*'* 50)
        print('')
        self.plot_boss_counts()
        plt.show()
        print('')
        print('*'* 50)
        print('')
        self.plot_skill_count()
        plt.show()
        print('')
        print('*'* 50)
        print('')
        self.print_boss_part()
        print('')
        print('*'* 50)
        print('')
        self.print_skill_per_dungeon()
        print('')
        print('*'* 50)
        print('')
        self.print_skill_per_boss()
        print('')
        print('*'* 50)
        print('')
