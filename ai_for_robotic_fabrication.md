
# AI For Robotic Fabrication Workshop: Reinforcement Learning for Intelligent Grid-Based Carving

**Team Members:**  
Sahil Yousaf, Kacper Wasilewski, Eleftheria Papadosifou, Nihal Ahmed M, Bhargava Sangam  
---

## Can an AI Agent Learn to Carve a Shape?

Inspired by the precision of traditional craftsmanship, especially techniques like Japanese joinery where every cut matters, we wanted to see if a machine could learn that same logic in a digital environment.

---

## The Idea

In fabrication, whether by hand or machine, material is often removed step by step — always with a goal in mind. But what if we gave an AI the freedom to decide how to remove that material?

To test this, we built a simple simulation:  
A square grid, filled in like a block of material. The AI’s task? Carve out a perfect circle from the square — one pixel at a time.

It sounds straightforward, but the catch is that the agent doesn’t know anything about geometry or circles. It only learns by trial and error, based on the rewards it gets for each move:

- **Remove unnecessary material** → good.  
- **Damage the desired shape** → bad.

---

## How It Works

We created a custom environment using Gymnasium and visualized the process with Pygame.  
The agent interacts with the grid:

- It selects a pixel to remove.  
- It gets rewarded if that move helps shape the circle.  
- It gets penalized if it removes something important.

To push the learning process, we ran the simulation over 1000 episodes, letting the AI fail, learn, and slowly improve its decisions.

---

## Conclusion

This experiment wasn’t about creating a perfect algorithm — it was about testing an idea:  
Can a machine learn to shape, not by following instructions, but by understanding the logic of the process?

By setting up a simple environment where an AI agent carves a circle out of a square, we opened up a space to explore how machines could eventually learn from their own actions, rather than relying on predefined rules. The results weren’t perfect — the agent made mistakes, tried random strategies, and often failed to find the optimal path.

But that’s what made it interesting.

The project became less about carving and more about the process of learning itself.  
It showed how complex, human-like decision-making can emerge from something as simple as trial, error, and feedback — and how that logic could one day feed into real-world fabrication systems, where robots don’t just follow orders, but learn how to make.

---

## What’s Next

This project opened up several directions to explore:

- Testing more advanced learning algorithms.  
- Scaling up to more complex shapes.  
- Connecting the digital learning process to real robotic arms and fabrication tools.

---

**Tags:**  
Material: Wood, Artificial Intelligence, Robotic Fabrication, Structure, Python, Adaptive Strategy
