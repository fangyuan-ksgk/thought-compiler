import React, { useState } from 'react';
import { Search, Edit, Check, X } from 'lucide-react';

const InteractiveArticleWriter = () => {
  const [step, setStep] = useState(0);
  const [topics, setTopics] = useState([]);
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [feedback, setFeedback] = useState('');

  const steps = [
    'Research',
    'Topic Selection',
    'Writing',
    'Review',
    'Publish'
  ];

  const mockFetchTopics = () => {
    // Simulating API call to fetch trending topics
    setTimeout(() => {
      setTopics([
        'The Impact of AI on Job Markets',
        'Climate Change: Recent Developments',
        'Advances in Quantum Computing',
        'The Future of Remote Work',
        'Blockchain Beyond Cryptocurrency'
      ]);
    }, 1500);
  };

  const handleTopicSelection = (topic) => {
    setSelectedTopic(topic);
    setStep(2); // Move to Writing step
  };

  const handleFeedbackSubmit = () => {
    // Here you would typically send the feedback to the AI
    console.log('Feedback submitted:', feedback);
    setFeedback('');
    setStep(4); // Move to Publish step
  };

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <div className="w-64 bg-white shadow-md p-6">
        <h2 className="text-xl font-bold mb-4">Progress</h2>
        <ul>
          {steps.map((s, index) => (
            <li key={s} className={`mb-2 ${index === step ? 'font-bold' : ''}`}>
              {s}
            </li>
          ))}
        </ul>
      </div>

      {/* Main content */}
      <main className="flex-1 p-6">
        {step === 0 && (
          <div>
            <h1 className="text-2xl font-bold mb-4">Research</h1>
            <p className="mb-4">AI is navigating the web to find trending topics...</p>
            <button 
              className="bg-blue-500 text-white px-4 py-2 rounded flex items-center"
              onClick={() => { mockFetchTopics(); setStep(1); }}
            >
              <Search className="mr-2" size={18} />
              Start Research
            </button>
          </div>
        )}
        {step === 1 && (
          <div>
            <h1 className="text-2xl font-bold mb-4">Topic Selection</h1>
            <ul>
              {topics.map(topic => (
                <li key={topic} className="flex justify-between items-center mb-2">
                  <span>{topic}</span>
                  <button 
                    className="bg-green-500 text-white px-3 py-1 rounded flex items-center"
                    onClick={() => handleTopicSelection(topic)}
                  >
                    <Check className="mr-1" size={16} />
                    Select
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
        {step === 2 && (
          <div>
            <h1 className="text-2xl font-bold mb-4">Writing</h1>
            <p className="mb-4">AI is writing an article on: {selectedTopic}</p>
            <button 
              className="bg-purple-500 text-white px-4 py-2 rounded flex items-center"
              onClick={() => setStep(3)}
            >
              <Edit className="mr-2" size={18} />
              Review Draft
            </button>
          </div>
        )}
        {step === 3 && (
          <div>
            <h1 className="text-2xl font-bold mb-4">Review</h1>
            <p className="mb-4">Please review the article and provide feedback:</p>
            <textarea
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              className="w-full p-2 mb-4 border rounded"
              placeholder="Enter your feedback here..."
              rows={4}
            />
            <button 
              className="bg-blue-500 text-white px-4 py-2 rounded flex items-center"
              onClick={handleFeedbackSubmit}
            >
              <Check className="mr-2" size={18} />
              Submit Feedback
            </button>
          </div>
        )}
        {step === 4 && (
          <div>
            <h1 className="text-2xl font-bold mb-4">Publish</h1>
            <p className="mb-4">Your article is ready to be published!</p>
            <button className="bg-green-500 text-white px-4 py-2 rounded flex items-center">
              <Check className="mr-2" size={18} />
              Publish Article
            </button>
          </div>
        )}
      </main>
    </div>
  );
};

export default InteractiveArticleWriter;