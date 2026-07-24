export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const { messages } = req.body;
  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: 'Invalid request body' });
  }

  const userQuestion = messages[messages.length - 1].content;

  try {
    // Step 1 — Embed with Pinecone inference API
    const embedRes = await fetch('https://api.pinecone.io/embed', {
      method: 'POST',
      headers: {
        'Api-Key': process.env.PINECONE_API_KEY,
        'Content-Type': 'application/json',
        'X-Pinecone-API-Version': '2024-10'
      },
      body: JSON.stringify({
        model: 'multilingual-e5-large',
        inputs: [{ text: userQuestion }],
        parameters: { input_type: 'query', truncate: 'END' }
      })
    });

    const embedData = await embedRes.json();
    const vector = embedData.data[0].values;

    // Step 2 — Query Pinecone
    const queryRes = await fetch(`https://${process.env.PINECONE_HOST}/query`, {
      method: 'POST',
      headers: {
        'Api-Key': process.env.PINECONE_API_KEY,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        vector: vector,
        topK: 5,
        namespace: 'portfolio',
        includeMetadata: true
      })
    });

    const queryData = await queryRes.json();
    const matches = queryData.matches || [];

    // Step 3 — Build context
    const context = matches
      .map((match, i) => {
        const filename = match.metadata?.filename || 'unknown';
        const text = match.metadata?.text || '';
        return `[Source ${i + 1}: ${filename}]\n${text}`;
      })
      .join('\n\n---\n\n');

    // Step 4 — Call Claude
    const claudeRes = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'x-api-key': process.env.ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'claude-haiku-4-5-20251001',
        max_tokens: 1024,
        system: `You are an assistant on Terry Migwi's portfolio website. Your job is to answer questions about Terry's projects only, using the retrieved context below.

If someone asks about anything outside of these projects, including Terry's personal background, availability, salary, or anything else not covered in the context, respond with: "I am sorry, my instructions are to stay within the scope of these projects. If you would like more information on this, feel free to contact Terry."

Important: Denta is a messaging assistant that works over both WhatsApp and SMS channels. Do not describe it as WhatsApp-only. Always refer to it as a messaging assistant.

Do not make up any information not present in the context. Keep responses concise and professional.

Portfolio overview:
This portfolio contains four projects:
1. Denta - a messaging assistant with RAG and tool calling for bookings, designed for any business taking bookings or appointments over a messaging channel.
2. Hybrid Search - a production-ready document search engine combining semantic search and keyword search with LLM answer synthesis, secured with JWT authentication.
3. TournamentIQ - a live AI match intelligence platform built on 2026 FIFA World Cup data, combining Elo ratings, Poisson probability modelling, Bayesian inference, and LangGraph orchestration to generate narrative match intelligence briefs.
4. Attendance Tracker - a Python automation that reduced weekly learner attendance tracking from 2 hours to under 10 minutes.

Retrieved context:
${context}`,
        messages: messages
      })
    });

    const claudeData = await claudeRes.json();

    if (!claudeRes.ok) {
      return res.status(claudeRes.status).json({ error: claudeData.error?.message || 'Claude API error' });
    }

    return res.status(200).json({ reply: claudeData.content[0].text });

  } catch (error) {
    console.error('Chat error:', error);
    return res.status(500).json({ error: error.message });
  }
}