// Run the frozen deployed BFF code against real inference HTTP fixture responses.
import { mapBodyRecommendations } from './recommendation.js';
import { snapshot, resolveBody, bodyRef } from './model.js';

let text = '';
for await (const chunk of process.stdin) text += chunk;
const input = JSON.parse(text);
const recs = mapBodyRecommendations(input.cut, input.catalog);
const policy = snapshot(input.preferences, input.catalog);
const people = input.cut.people.map(person => {
  const recommendation = recs.get(person.index);
  const manual = input.catalog.assets.find(a => a.characterId === input.manualCharacterId);
  return resolveBody(person.index, policy, recommendation, input.catalog,
    person.candidates.map(c => c.pose_id), input.intent ?? 'inherit',
    manual ? bodyRef(manual) : null, input.revision ?? 0);
});
process.stdout.write(JSON.stringify({ recommendations: [...recs.values()], people }));
