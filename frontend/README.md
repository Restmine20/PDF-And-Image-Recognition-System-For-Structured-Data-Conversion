# RecognitionSystem — Frontend

Веб-интерфейс для системы распознавания таблиц из PDF и изображений.

Стек: **React 18 + TypeScript + Vite + React Router + CSS Modules**.
Управление состоянием — **React Context**, сессия хранится в `localStorage`.

## Запуск

```bash
# из этой папки
rm -rf dist node_modules/.cache    # на всякий случай чистим старое
npm install
cp .env.example .env
npm run build
npm run dev  # dev-сервер на http://localhost:5173

docker build --build-arg VITE_USE_MOCKS=true -t recsys-frontend . # режим с моками

docker run --rm -p 8080:80 --add-host backend:127.0.0.1 recsys-frontend
```

Запросы на `/api/*` проксируются на `http://localhost:8000`

### Запуск без бэкенда
Для запуска чисто frontend части:
```
VITE_USE_MOCKS=true
```


## Структура

```
src/api -> обёртки над Fetch API
src/components -> UI-блоки и CSS-модули
src/context -> AuthContext и ToastContext
src/hooks-> валидация файлов и так далее
src/pages -> страницы маршрутизатора
src/styles -> variables.css и global.css
src/types -> интерфейсы TypeScript
```

## Странички
```
/ -> Главная страница с drag&drop загрузкой файлов и предпросмотром перед отправкой
/auth -> Регистрация и вход
/processing/:jobId -> Процесс обработки
/result/:jobId -> скачивание рузльтата
/history -> Доступна только авторизованным! История загрузок
/api-key -> Доступна только авторизованным! Генерация нового API-ключа (показывается один раз, старый ключ аннулируется)
```
