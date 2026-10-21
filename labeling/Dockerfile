FROM node:22-alpine
ENV NODE_ENV=production HOST=0.0.0.0 PORT=8080 LABEL_LOCAL_STORE=/data/state.json LABEL_BACKUP_DIR=/data/backups
WORKDIR /app
COPY --chown=node:node package.json server.mjs ./
COPY --chown=node:node api ./api
COPY --chown=node:node lib ./lib
COPY --chown=node:node data/seed.json ./data/seed.json
COPY --chown=node:node tools ./tools
COPY --chown=node:node index.html app.js app.css ./
COPY --chown=node:node assets ./assets
RUN mkdir -p /data && chown node:node /data
USER node
EXPOSE 8080
CMD ["node", "server.mjs"]
