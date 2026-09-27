#!/usr/bin/env bash
# Создаёт роль, базу и схему для хранилища проектов.
#
# Идемпотентен: повторный запуск ничего не ломает. Пароль роли генерируется
# один раз и переиспользуется из ~/.config/ai-workspace/env.
#
# Требует SUDO_PW в окружении (пароль для sudo), чтобы не спрашивать
# интерактивно.
set -euo pipefail

ENV_FILE="$HOME/.config/ai-workspace/env"
SCHEMA_FILE="$HOME/projects/ai-farm/sql/schema.sql"
DB_NAME="aiworkspace"
DB_ROLE="rag"

# -p '' убирает приглашение "[sudo] password for ...", которое печатается без
# перевода строки и склеивается со следующим сообщением скрипта.
sudo_run() { printf '%s\n' "${SUDO_PW:?SUDO_PW не задан}" | sudo -S -p '' "$@"; }

mkdir -p "$(dirname "$ENV_FILE")"

# Пароль роли: берём существующий или создаём новый.
if [ -f "$ENV_FILE" ] && grep -q '^RAG_DB_PASSWORD=' "$ENV_FILE"; then
    DB_PASSWORD=$(grep '^RAG_DB_PASSWORD=' "$ENV_FILE" | cut -d= -f2-)
    echo "пароль роли взят из $ENV_FILE"
else
    DB_PASSWORD=$(openssl rand -hex 20)
    echo "сгенерирован новый пароль роли"
fi

# Роль
if [ "$(sudo_run -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='${DB_ROLE}'")" = "1" ]; then
    sudo_run -u postgres psql -q -c "ALTER ROLE ${DB_ROLE} LOGIN PASSWORD '${DB_PASSWORD}';"
    echo "роль ${DB_ROLE}: пароль обновлён"
else
    sudo_run -u postgres psql -q -c "CREATE ROLE ${DB_ROLE} LOGIN PASSWORD '${DB_PASSWORD}';"
    echo "роль ${DB_ROLE}: создана"
fi

# База
if [ "$(sudo_run -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'")" = "1" ]; then
    echo "база ${DB_NAME}: уже существует"
else
    sudo_run -u postgres createdb -O "${DB_ROLE}" "${DB_NAME}"
    echo "база ${DB_NAME}: создана"
fi

# Схема. CREATE EXTENSION требует суперпользователя, поэтому применяем от
# postgres. Домашний каталог пользователя postgres недоступен, а stdin занят
# паролем для sudo -S, поэтому на время работы выкладываем копию в /tmp.
TMP_SCHEMA=$(mktemp /tmp/rag-schema.XXXXXX.sql)
install -m 644 "$SCHEMA_FILE" "$TMP_SCHEMA"
trap 'rm -f "$TMP_SCHEMA"' EXIT
sudo_run -u postgres env PGOPTIONS='-c client_min_messages=warning' \
    psql -q -v ON_ERROR_STOP=1 -d "${DB_NAME}" -f "$TMP_SCHEMA"
echo "схема применена"

# Права для рабочей роли. Каждый оператор отдельным -c: stdin нужен sudo.
sudo_run -u postgres psql -q -v ON_ERROR_STOP=1 -d "${DB_NAME}" \
    -c "GRANT ALL ON SCHEMA public TO ${DB_ROLE};" \
    -c "GRANT ALL ON ALL TABLES IN SCHEMA public TO ${DB_ROLE};" \
    -c "GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO ${DB_ROLE};" \
    -c "GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO ${DB_ROLE};" \
    -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO ${DB_ROLE};" \
    -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO ${DB_ROLE};"
echo "права выданы"

# Файл подключения для индексатора и MCP-сервера.
umask 077
cat > "$ENV_FILE" <<ENV
# Подключение к хранилищу проектов. Права 600 — читается только владельцем.
RAG_DB_HOST=127.0.0.1
RAG_DB_PORT=5432
RAG_DB_NAME=${DB_NAME}
RAG_DB_USER=${DB_ROLE}
RAG_DB_PASSWORD=${DB_PASSWORD}
ENV
chmod 600 "$ENV_FILE"
echo "подключение записано в $ENV_FILE (права 600)"

# Проверка, что роль действительно может подключиться и видеть таблицы.
echo "--- проверка ---"
PGPASSWORD="$DB_PASSWORD" psql -h 127.0.0.1 -U "${DB_ROLE}" -d "${DB_NAME}" -tAc \
    "SELECT 'таблиц: ' || COUNT(*) FROM information_schema.tables WHERE table_schema='public';"
PGPASSWORD="$DB_PASSWORD" psql -h 127.0.0.1 -U "${DB_ROLE}" -d "${DB_NAME}" -tAc \
    "SELECT 'pgvector: ' || extversion FROM pg_extension WHERE extname='vector';"
