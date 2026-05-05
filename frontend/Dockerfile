FROM node:20-alpine AS builder
WORKDIR /app

COPY package.json package-lock.json* ./

RUN (npm ci --no-audit --no-fund --include=optional) \
    || (echo ">> npm ci failed, falling back to npm install" \
        && rm -f package-lock.json \
        && npm install --no-audit --no-fund)

COPY . .

ARG VITE_API_BASE_URL=""
ARG VITE_USE_MOCKS="false"
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
ENV VITE_USE_MOCKS=$VITE_USE_MOCKS

RUN npm run build

FROM nginx:1.27-alpine AS runtime

RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/recsys.conf

COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD wget -q --spider http://localhost/healthz || exit 1

CMD ["nginx", "-g", "daemon off;"]
