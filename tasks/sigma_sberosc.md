Чтобы настроить установку Python-пакетов из индекса SberOSC (Sigma), выполните шаги:

1. Откройте `sberosc.sigma.sbrf.ru` и авторизуйтесь.
2. Перейдите в `Profile` и сгенерируйте токен (далее `TOKEN`).

После этого настройте `pip` одним из способов:

- через конфигурационный файл `pip.conf` (на Windows - `pip.ini`)
- через переменные окружения `PIP_INDEX_URL` и `PIP_TRUSTED_HOST`

Пример конфигурации:

```ini
[global]
index-url = https://token:TOKEN@sberosc.sigma.sbrf.ru/pypi/simple
trusted-host = sberosc.sigma.sbrf.ru
```

Где разместить файл конфигурации `pip`:

- SberOS/Linux: `~/.config/pip/pip.conf` или `~/.pip/pip.conf`
- macOS: `~/.pip/pip.conf`
- Windows: `%APPDATA%\pip\pip.ini`
