def split_into_syllables_simple(word: str) -> list[str]:
    """
    Разбивает русское слово на слоги.
    Правило: группы согласных обычно не делятся, кроме удвоенных.
    """
    vowels = 'аеёиоуыэюя'
    word = word.lower().strip()

    if not word:
        return []

    syllables = []
    i = 0

    while i < len(word):
        current_syllable = ''

        # Собираем согласные в начале слога
        while i < len(word) and word[i] not in vowels:
            current_syllable += word[i]
            i += 1

        # Добавляем гласную
        if i < len(word):
            current_syllable += word[i]
            i += 1

            # Смотрим что дальше
            if i < len(word):
                # Если следующая гласная - завершаем слог
                if word[i] in vowels:
                    syllables.append(current_syllable)
                    continue

                # Есть согласные - проверяем удвоенные ли они
                j = i
                while j < len(word) and word[j] not in vowels:
                    j += 1

                # Если дальше нет гласных - берём все
                if j >= len(word):
                    current_syllable += word[i:]
                    syllables.append(current_syllable)
                    break

                # Проверяем, есть ли удвоенные согласные
                consonants = word[i:j]
                doubled_pos = -1
                for k in range(len(consonants) - 1):
                    if consonants[k] == consonants[k + 1]:
                        doubled_pos = k
                        break

                if doubled_pos >= 0:
                    # Есть удвоенные - берём до первого из пары
                    current_syllable += consonants[: doubled_pos + 1]
                    i += doubled_pos + 1
                    syllables.append(current_syllable)
                else:
                    # Нет удвоенных - все согласные к следующему слогу
                    syllables.append(current_syllable)
            else:
                syllables.append(current_syllable)

    return syllables


if __name__ == '__main__':
    words = ['программа', 'аист', 'страна', 'обезьяна', 'поющая', 'компьютер']
    for w in words:
        print('{:15} -> {}'.format(w, '-'.join(split_into_syllables_simple(w))))

    # Вывод:
    # программа      -> про-грам-ма
    # аист           -> а-ист
    # страна         -> стра-на
    # обезьяна       -> о-бе-зья-на
    # поющая         -> по-ю-ща-я
    # компьютер      -> ко-мпью-тер  // некорректно, но для подсчета слогов сгодится
