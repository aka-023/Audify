export async function sendGeminiRequest(
  apiConfig,
  standardMessages,
  mimeType
) {
  const connectionMethod = apiConfig.cloudApiMethod || 'direct';

  if (!apiConfig.cloudModelName) throw new Error('Cloud Gemini model name not set.');

  let endpoint;
  let headers = { 'Content-Type': 'application/json' };
  const modelToUse = apiConfig.cloudModelName;
  
  if (connectionMethod === 'proxy') {
    if (!apiConfig.cloudProxyUrl) throw new Error('API Gateway Endpoint not configured for Vertex AI (GCP) method.');
    if (!apiConfig.gcpApiKey) throw new Error('GCP API Key not configured for Vertex AI (GCP) method.');
    endpoint = apiConfig.cloudProxyUrl;
    headers['x-api-key'] = apiConfig.gcpApiKey;
    headers['X-Model-Name'] = modelToUse;
  } else {
    if (!apiConfig.cloudApiKey) throw new Error('Cloud Gemini API Key not configured.');
    const apiKey = apiConfig.cloudApiKey;
    const cloudBaseUrl = 'https://generativelanguage.googleapis.com/v1beta/models/';
    const fullApiUrl = `${cloudBaseUrl}${modelToUse}:generateContent`;
    endpoint = `${fullApiUrl}?key=${apiKey}`;
  }

  // helper to build Gemini request body from message array
  function buildPayload(messages) {
    let systemInstruction = null;
    const geminiContents = [];
    const messagesToProcess = [...messages];

    if (messagesToProcess.length > 0 && messagesToProcess[0].role === 'system') {
      const systemMsg = messagesToProcess.shift();
      if (systemMsg.content) {
        systemInstruction = { parts: [{ text: systemMsg.content }] };
      }
    }

    for (const message of messagesToProcess) {
      const role = message.role === 'assistant' ? 'model' : 'user';
      const parts = [];

      if (message.content) {
        parts.push({ text: message.content });
      }

      if (message.images && message.images.length > 0 && role === 'user') {
        for (const imgData of message.images) {
          parts.push({
            inline_data: {
              mime_type: mimeType,
              data: imgData
            }
          });
        }
      }

      if (parts.length > 0) {
        geminiContents.push({ role, parts });
      }
    }

    if (geminiContents.length > 0 && geminiContents[0].role === 'model') {
      geminiContents.shift();
    }

    if (geminiContents.length === 0) {
      throw new Error(`Cannot send empty request to Gemini via ${connectionMethod} method.`);
    }

    const geminiPayload = {
      contents: geminiContents,
      ...(systemInstruction && { systemInstruction: systemInstruction })
    };

    const modelsWithThinkingConfig = ['gemini-2.5-flash', 'gemini-2.5-flash-lite'];
    if (modelsWithThinkingConfig.includes(modelToUse)) {
      geminiPayload.generationConfig = {
        thinkingConfig: { thinkingBudget: -1 }
      };
    }

    return JSON.stringify(geminiPayload);
  }

  // helper to POST to Gemini and extract text result (keeps original error handling)
  async function callGemini(body) {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: headers,
      body: body
    });

    let data;
    try {
      data = await response.json();
    } catch (e) {
      const errorBodyText = await response.text();
      throw new Error(`API Error (${response.status} ${response.statusText}). Response: ${errorBodyText.substring(0, 200)}`);
    }

    if (!response.ok) {
      let detailedError = `API Error via ${connectionMethod} (${response.status} ${response.statusText})`;
      if (data && data.error) {
        if (typeof data.error === 'string') {
          detailedError += `: ${data.error}`;
        } else if (data.error.message) {
          detailedError += `: ${data.error.message}`;
        } else {
          detailedError += `. Response: ${JSON.stringify(data.error).substring(0, 200)}`;
        }
      } else if (data) {
        detailedError += `. Response: ${JSON.stringify(data).substring(0, 200)}`;
      }
      throw new Error(detailedError);
    }

    let responseText = `Error: Could not parse ${connectionMethod === 'proxy' ? 'proxied' : ''} Gemini response.`;
    try {
      if (data.candidates && data.candidates[0]?.content?.parts) {
        const textParts = data.candidates[0].content.parts.filter(part => part.text);
        if (textParts.length > 0) {
          responseText = textParts.map(p => p.text).join('');
        }
      } else if (data.promptFeedback?.blockReason) {
        responseText = `Request blocked by API: ${data.promptFeedback.blockReason}`;
        if (data.promptFeedback.safetyRatings) {
          responseText += ` - Details: ${data.promptFeedback.safetyRatings.map(r => `${r.category}: ${r.probability}`).join(', ')}`;
        }
      } else if (data.candidates && data.candidates[0]?.finishReason && data.candidates[0].finishReason !== "STOP") {
        responseText = `Request finished unexpectedly. Reason: ${data.candidates[0].finishReason}`;
        const safetyRatingsInfo = data.candidates[0].safetyRatings?.map(r => `${r.category}: ${r.probability}`).join(', ');
        if (safetyRatingsInfo) responseText += ` (Safety Ratings: ${safetyRatingsInfo})`;
        if (data.candidates[0].content?.parts?.some(p => p.text)) {
          const partialText = data.candidates[0].content.parts.filter(p => p.text).map(p => p.text).join('');
          responseText += `\nPartial content: ${partialText}`;
        }
      } else if (data.error) {
        responseText = `${connectionMethod === 'proxy' ? 'Proxied ' : ''}Gemini API Error: ${data.error.message || 'Unknown error'}`;
      }
    } catch (parseError) {
      responseText = `Error: Failed to process ${connectionMethod === 'proxy' ? 'proxied ' : ''}Gemini response content.`;
    }
    return responseText;
  }

  try {
    // --- IMPROVED: Clarify user's prompt once, using more detailed examples & rules ---
    const firstUserIndex = (standardMessages || []).findIndex(m => m && m.role === 'user' && typeof m.content === 'string' && m.content.trim().length > 0);

    let messagesForFinalRequest = standardMessages;

    if (firstUserIndex !== -1) {
      const userInput = standardMessages[firstUserIndex].content;

      // Stronger, example-driven clarification instruction:
      const clarificationPrompt = [
        {
          role: 'system',
          content:
`You are a task clarifier whose ONLY job is to rewrite a user's informal or vague request into a single, clear, specific, and actionable instruction an automated AI can perform. Follow these strict rules and then output ONLY the rewritten instruction (no explanation, no extra text):

1) Output must be one concise imperative sentence describing exactly the action and target (e.g., "Open youtube.com and search for 'lofi beats'." or "Like the current playing video on YouTube.").  
2) Turn vague words into explicit referents: "this/that" -> "the current playing video", "similar" -> "other videos similar to the current playing video", etc.  
3) If the user mentions a service (YouTube, Gmail, Twitter), use an appropriate action: "open <site>", "search <site> for: <terms>", "like the current playing video", "subscribe to <channel name>".  
4) Preserve URLs, filenames, numbers, languages, and any modifiers (e.g., "only", "top 5", "in Hindi"). Include them in the rewritten task.  
5) If the user says something like "search for youtube" interpret it as the intent to open/search YouTube and produce a direct action ("Open youtube.com" or "Search YouTube for: <terms>" if terms are present).  
6) If ambiguous, choose the most likely specific interpretation (do NOT ask follow-up questions) and be concrete.  
7) Do NOT add commentary, reasoning, or multiple options — provide exactly one instruction line.

Examples:
- User: "search for youtube"  ->  Rewritten: "Open youtube.com."
- User: "like this video"    ->  Rewritten: "Like the current playing video on YouTube."
- User: "find me top tutorials" -> Rewritten: "Search YouTube for 'top tutorials' and return the top 5 results."
- User: "send email to john" -> Rewritten: "Compose and send an email to john@example.com with the subject and body specified by the user." (if no details, assume typical defaults)

Now rewrite the following ORIGINAL user request into one clear, executable task:
`
        },
        { role: 'user', content: `ORIGINAL: ${userInput}` }
      ];

      const clarifyBody = buildPayload(clarificationPrompt);
      const clarifiedTask = await callGemini(clarifyBody);

      if (typeof clarifiedTask === 'string' && clarifiedTask.trim().length > 0) {
        messagesForFinalRequest = standardMessages.map((m, idx) => {
          if (idx === firstUserIndex) {
            return { ...m, content: clarifiedTask };
          }
          return m;
        });
      }
    }

    // --- Now proceed exactly as before with the (possibly updated) messagesForFinalRequest ---
    const finalBody = buildPayload(messagesForFinalRequest);
    const finalResult = await callGemini(finalBody);

    return finalResult;
  } catch (error) {
    if (error instanceof Error && error.message.startsWith('API Error')) {
      throw error;
    } else {
      throw new Error(`Failed to communicate with ${connectionMethod} endpoint or process response: ${error.message}`);
    }
  }
}
