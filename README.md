# rag-sales-mistral
Sistema RAG con Mistral-7B-Instruct-v0.3 para análisis de ventas en lenguaje natural. Chat interactivo sobre dataset CSV usando búsqueda semántica y API de Mistral. Ejecutable en Google Colab.

## Respuestas proveídas
### 1
 Tú: Ordenes que hay 
🤖 [rag] Según el contexto proporcionado, las órdenes registradas son:

1. **Orden #0.5** (Classic Cars):
   - Monto: $0.37
   - Estado: Shipped
   - País: Australia
   - Fecha: 2004-07-19
   - Tamaño: Medium
   - Cliente: Souveniers And Things Co.

2. **Orden #0.5** (Classic Cars):
   - Monto: $0.47
   - Estado: Shipped
   - País: Australia
   - Fecha: 2004-07-19
   - Tamaño: Medium
   - Cliente: Souveniers And Things Co.

3. **Orden #0.5** (Classic Cars):
   - Monto: $0.40
   - Estado: Shipped
   - País: Australia
   - Fecha: 2004-07-19
   - Tamaño: Medium
   - Cliente: Souveniers And Things Co.

4. **Orden #0.5** (Vintage Cars):
   - Monto: $0.21
   - Estado: Shipped
   - País: Australia
   - Fecha: 2004-07-19
   - Tamaño: Small
   - Cliente: Souveniers And Things Co.
### 2
🔍 Tú: Dame todos los clientes
🤖 [rag] Según el contexto proporcionado, el único cliente registrado es:

- **Souveniers And Things Co.**
  - Venta #0.5: $0.07 (Tamaño: Small)
  - Venta #0.5: $0.38 (Tamaño: Medium)
  - Venta #0.5: $0.38 (Tamaño: Medium)
  - Venta #0.5: $0.23 (Tamaño: Small)
### 3

🔍 Tú: quiero el monto mas grande
🤖 [rag] El monto más grande registrado en el contexto es **$0.47** (VENTA #0.5).

**Nota:** Todas las órdenes tienen el mismo número (#0.5), lo que sugiere que podrían ser transacciones parciales o registros duplicados del mismo evento. No hay suficiente información para determinar si son órdenes independientes o parte de un mismo proceso.
