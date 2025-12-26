Authors: Julia Wiesinger, Patrick Marlow and Vladimir Vuskovic 
#### Acknowledgements
Content contributors
Evan Huang
Emily Xue
Olcan Sercinoglu
Sebastian Riedel
Satinder Baveja
Antonio Gulli
Anant Nawalgaria
#### Curators and Editors
Antonio Gulli
Anant Nawalgaria
Grace Mollison
#### Technical Writer
Joey Haymaker
#### Designer
Michael Lanning

![[22365_19_Agents_v8_images/page_3_img_1.png]]


> [!NOTE]
> 这种将 reasoning、logic 以及外部信息访问与 Generative AI 模型相结合的组合，引入了 Agent 的概念。

# **介绍** 

人类在处理复杂的模式识别任务方面表现出色。然而，在得出结论之前，人类通常依赖工具——如书籍、Google Search 或计算器——来补充其先验知识。正如人类一样，Generative AI 模型也可以经过训练来使用工具，从而获取实时信息或建议现实世界的行动。 例如，模型可以利用数据库检索工具来访问特定信息（如客户的购买记录），以便生成个性化的购物建议。或者，根据用户的查询，模型可以发起各种 API 调用，向同事发送电子邮件回复，或代表用户完成金融交易。 为实现这一点，模型不仅必须能够访问一组外部工具，还需要具备以自主方式规划和执行任务的能力。这种将 reasoning、logic 以及外部信息访问与 Generative AI 模型相结合的组合，引入了 Agent 的概念，即一种超越了 Generative AI 模型独立能力的程序。本白皮书将详细探讨这些及相关的各个方面。

# **什么是Agent？**

从最基本的层面来说，生成式人工智能（Geneative AI）智能体可以定义为一种应用程序，它通过观察世界并利用其可用工具采取行动来实现目标。智能体具有自主性，可以独立于人类干预而行动，尤其是在被赋予明确的目标或目的时。智能体还可以主动地朝着目标前进。即使没有来自人类的明确指令，智能体也能推理出下一步应该做什么才能实现最终目标。虽然人工智能中的智能体概念相当宽泛且强大，但本白皮书重点关注截至发布时生成式人工智能模型能够构建的特定类型的智能体。

为了理解智能体的内部运作机制，我们首先来介绍驱动智能体行为、动作和决策的基础组件。这些组件的组合可以描述为认知架构，通过对这些组件进行组合搭配，可以实现多种不同的认知架构。着眼于核心功能，智能体的认知架构包含三个基本组件，如图 1 所示。

![[page_6_img_1.png]]
![][image1]  
图 1\. 通用代理架构和组件

#### **模型**

在 **agent** 的范畴内，**model** 指的是将被用作 **agent** 流程核心决策者的 **language model (LM)**。一个 **agent** 使用的 **model** 可以是一个或多个任何规模（small / large）的 **LM**，只要它们能够遵循基于指令的 **reasoning** 和 **logic** 框架，例如 **ReAct**、**Chain-of-Thought** 或 **Tree-of-Thoughts**。 根据特定 **agent architecture** 的需求，**models** 可以是通用型、**multimodal** 或经过 **fine-tuned** 的。为了获得最佳的 **production** 结果，您应当利用最适合所需终端应用的 **model**，且理想情况下，该 **model** 应当在与您计划在 **cognitive architecture** 中使用的工具相关联的 **data signatures** 上进行过训练。 需要注意的是，**model** 通常并非针对 **agent** 的特定 **configuration settings**（即工具选择、**orchestration**/**reasoning** 设置）进行训练的。然而，通过向其提供展示 **agent** 能力的示例（包括 **agent** 在各种语境下使用特定工具或 **reasoning** 步骤的实例），可以针对 **agent** 的任务进一步优化（refine）该 **model**。

**工具**

**Foundational models** 尽管在文本和图像生成方面表现出色，但由于无法与外部世界交互，其能力仍然受到限制。**Tools** 弥补了这一差距，赋能 **agents** 与外部数据和服务进行交互，同时解锁了远超底层模型单一能力的更广泛的操作范围。 **Tools** 可以表现为多种形式，并具有不同程度的复杂性，但通常与常见的 **Web API** 方法（如 **GET**、**POST**、**PATCH** 和 **DELETE**）相对应。例如，一个 **tool** 可以更新数据库中的客户信息，或获取天气数据以影响 **agent** 向用户提供的旅行建议。 借助 **tools**，**agents** 能够访问并处理现实世界的信息。这使它们能够支持更专业的系统，例如 **retrieval augmented generation (RAG)**，从而显著扩展了 **agent** 的能力，使其超越了 **foundational model** 独立实现的效果。我们将在下文中详细讨论 **tools**，但最核心的一点是：**tools** 桥接了 **agent** 内部能力与外部世界之间的鸿沟，开启了更广阔的可能性。

# **编排层**

**Orchestration layer** 描述了一个循环过程，该过程支配着 **agent** 如何接收信息、进行内部 **reasoning**，并利用该 **reasoning** 为其下一个动作或决策提供依据。通常情况下，这一循环将持续进行，直到 **agent** 达成目标或触及停止点。 **Orchestration layer** 的复杂程度会根据 **agent** 及其所执行的任务而有所不同。有些循环可能只是带有决策规则的简单计算，而另一些则可能包含链式逻辑，涉及额外的 **machine learning** 算法，或应用其他 **probabilistic reasoning** 技术。我们将在 **cognitive architecture** 部分进一步详细探讨 **agent orchestration layers** 的具体实现。

# **Agents vs. models**

为了更清楚地理解代理和模型之间的区别，请参考以下图表：

| Models                                                                                                                                                    | Agents                                                                                                                                                                                                     |
| --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 他们的知识仅限于训练数据中可用的内容。                                                                                                                                       | 知识是通过工具与外部系统连接而扩展的                                                                                                                                                                                         |
| 基于用户查询（user query）进行的单次 **inference**（推理）或预测。除非专门为该 **model** 实现了相关功能，否则不具备对 **session history** 或持续 **context**（即 **chat history**）的管理能力。                | 管理 **session history**（即 **chat history**），以允许基于用户查询以及 **orchestration layer** 中所做出的决策进行多轮（**multi turn**）**inference** / 预测。在此语境下，“**turn**”被定义为交互系统与 **agent** 之间的一次交互（即：1 次传入的事件/查询和 1 次 **agent** 响应）。 |
| 没有原生工具实现。                                                                                                                                                 | 工具已在代理架构中原生实现。                                                                                                                                                                                             |
| 未实现原生的 **logic layer**。用户可以将 **prompts** 构建为简单的问题，或者使用 **reasoning frameworks**（如 **CoT**、**ReAct** 等）来构建复杂的 **prompts**，以引导 **model** 进行 **prediction**。 | 原生 **cognitive architecture**，使用 **CoT**、**ReAct** 等 **reasoning frameworks**，或其他如 **LangChain** 的预构建 **agent frameworks**。                                                                                |

# **认知架构：智能体如何运作**

想象一位在繁忙厨房里的厨师。他们的目标是为餐厅顾客烹制美味佳肴，这涉及 **planning**（规划）、**execution**（执行）和 **adjustment**（调整）的循环：

* 他们收集信息，例如顾客的点单以及储藏室和冰箱里有哪些食材。 
* 他们根据刚收集的信息，对可以制作哪些菜肴和风味组合进行内部 **reasoning**。 
* 他们采取行动制作菜肴：切菜、调制香料、煎肉。

在过程的每个阶段，厨师会根据需要进行调整，随着食材的消耗或收到客户反馈来优化他们的计划，并利用之前的结果序列来确定下一个行动计划。这种信息摄入、**planning**、执行和调整的循环，描述了厨师为达成目标而采用的一种独特的 **cognitive architecture**（认知架构）。

就像厨师一样，**agents** 可以通过迭代处理信息、做出明智决策并根据之前的输出优化后续行动，利用 **cognitive architectures** 来实现最终目标。**agent cognitive architectures** 的核心是 **orchestration layer**（编排层），负责维护 **memory**（记忆）、**state**（状态）、**reasoning** 和 **planning**。它利用快速发展的 **prompt engineering** 领域及相关框架来引导 **reasoning** 和 **planning**，使 **agent** 能够更有效地与其环境交互并完成任务。关于语言模型的 **prompt engineering** 框架和任务规划领域的研究正在迅速发展，产生了多种极具前景的方法。虽然未详尽列举，但以下是本白皮书发布时最流行的一些框架和 **reasoning** 技术：

* **ReAct**：一种 **prompt engineering** 框架，它为语言模型提供了一种思维过程策略，使其能够针对用户查询进行 **Reason**（推理）并采取行动（**Act**），无论是否存在上下文示例（in-context examples）。**ReAct** 提示已证明优于多个 **SOTA** 基准，并提高了 **LLMs** 的人类互操作性和可信度。 
* **Chain-of-Thought (CoT)**：一种 **prompt engineering** 框架，通过中间步骤实现 **reasoning** 能力。**CoT** 有多种子技术，包括 **self-consistency**、**active-prompt** 和 **multimodal CoT**，每种技术根据具体应用场景各有优劣。 
* **Tree-of-thoughts (ToT)**：一种非常适合探索或战略性前瞻任务的 **prompt engineering** 框架。它对 **chain-of-thought** 提示进行了泛化，允许模型探索各种思维链，这些思维链作为利用语言模型解决通用问题的中间步骤。

**Agents** 可以利用上述 **reasoning** 技术之一，或许多其他技术，来为给定的用户请求选择最佳的下一步行动。例如，让我们考虑一个被编程为使用 **ReAct** 框架为用户查询选择正确行动和工具的 **agent**。事件序列可能如下所示：

1. 用户向 **agent** 发送查询。 
2. **Agent** 开始 **ReAct** 序列。 
3. **Agent** 向模型提供一个 **prompt**，要求其生成下一个 **ReAct** 步骤之一及其对应的输出： 
	* a. **Question**：来自用户查询的输入问题，随 **prompt** 一起提供。
	* b. **Thought**：模型关于下一步该做什么的思考。 
	* c. **Action**：模型关于采取什么行动的决策。 
		* i. 这是进行 **tool** 选择的地方。 
		* ii. 例如，一个 **action** 可以是 [Flights, Search, Code, None] 之一，其中前三个代表模型可以选择的已知 **tool**，
	- d. **Action input**：模型决定为工具提供哪些输入（如果有）。
	* e. **Observation**：**action** / **action input** 序列的结果。 
		* i. 这种 **thought** / **action** / **action input** / **observation** 可以根据需要重复 N 次。 
	* f. **Final answer**：模型对原始用户查询的最终回答。
4. **ReAct** 循环结束，最终答案被返回给用户。

![[page_11_img_1.png]]![][image2]图 2\. 编排层中使用 ReAct 推理的示例代理

如图 2 所示，**model**、**tools** 和 **agent configuration** 协同工作，基于用户的原始 **query** 向其提供一个 **grounded**（有根据的）且简洁的 **response**。尽管 **model** 本可以基于其先验知识猜测一个答案（即 **hallucinated**，产生幻觉），但它转而使用了一个 **tool (Flights)** 来搜索实时的外部信息。这些额外信息被提供给 **model**，使其能够基于真实的客观数据做出更明智的决策，并将这些信息总结后反馈给用户。

  
总结而言，**agent** 响应的质量直接取决于 **model** 对各项任务进行 **reasoning** 和 **act** 的能力，包括选择正确 **tools** 的能力，以及这些 **tools** 定义的完善程度。正如厨师利用新鲜食材并关注客户反馈来精心烹制菜肴一样，**agents** 依靠严密的 **reasoning** 和可靠的信息来交付最佳结果。在下一章节中，我们将深入探讨 **agents** 与新鲜数据（fresh data）连接的各种方式。


# **工具：我们通往外部世界的钥匙**
虽然语言模型擅长处理信息，但它们缺乏直接感知和影响现实世界的能力。这限制了它们在需要与外部系统或数据交互的情境中的效用。这意味着，在某种意义上，一个语言模型的能力上限取决于它从训练数据中所学到的内容。但无论我们向模型投入多少数据，它们仍然缺乏与外部世界交互的基本能力。那么，我们该如何赋能模型，使其能够与外部系统进行实时的、具备 **context-aware**（上下文感知）能力的交互呢？**Functions**、**Extensions**、**Data Stores** 和 **Plugins** 都是向模型提供这一关键能力的方式。

尽管它们有许多不同的称呼，但 **tools** 正是建立 **foundational models** 与外部世界之间联系的纽带。这种与外部系统和数据的连接，使我们的 **agent** 能够执行更广泛的任务，并具有更高的准确性和可靠性。例如，**tools** 可以使 **agents** 能够根据特定的指令集调整智能家居设置、更新日历、从数据库中获取用户信息或发送电子邮件。

截至本白皮书发布之日，**Google** 模型能够交互的主要工具类型有三种：**Extensions**、**Functions** 和 **Data Stores**。通过为 **agents** 配备 **tools**，我们释放了巨大的潜力，使它们不仅能够理解世界，还能对世界采取行动，从而为无数新的应用和可能性开启了大门。

# **扩展**

理解 **Extensions** 最简单的方式是将它们视为以标准化方式桥接 **API** 与 **agent** 之间鸿沟的工具，使 **agents** 能够无缝执行 **APIs**，而无需考虑其底层实现。假设您构建了一个目标是帮助用户预订机票的 **agent**。您知道自己想要使用 **Google Flights API** 来获取航班信息，但不确定如何让您的 **agent** 调用该 **API endpoint**。

![[page_13_img_1.png]]![][image3]图 3\. 代理如何与外部 API 交互？

一种方法是实现 **custom code** 来接收传入的 **user query**，解析查询中的相关信息，然后发起 **API call**。 例如，在 **flight booking**（航班预订）用例中，用户可能会说：“我想预订从 Austin 到 Zurich 的航班。”在这种情况下，我们的 **custom code** 方案需要在尝试发起 **API call** 之前，从 **user query** 中提取出 “Austin” 和 “Zurich” 作为相关实体（entities）。 但如果用户说“我想预订去 Zurich 的航班”，却从未提供出发城市，会发生什么呢？由于缺少必要数据，**API call** 将会失败，为了处理此类 **edge cases**（边缘情况）和 **corner cases**（极端情况），还需要编写更多代码。这种方法不具备可扩展性（not **scalable**），并且在任何超出已实现的 **custom code** 逻辑的场景下都很容易出错。



一种更具鲁棒性（resilient）的方法是使用 **Extension**。**Extension** 通过以下方式桥接 **agent** 与 **API** 之间的鸿沟： 
1. 通过示例教导 **agent** 如何使用 **API endpoint**。 
2. 教导 **agent** 为了成功调用 **API endpoint**，需要哪些 **arguments** 或 **parameters**。
![[page_14_img_1.png]]![][image4]图 4\. 扩展程序将代理连接到外部 API

**Extensions** 可以独立于 **agent** 构建，但应作为 **agent** 配置的一部分提供。在 **run time**，**agent** 利用 **model** 和示例来决定哪个 **Extension**（如果有的话）适合解决 **user query**。这突显了 **Extensions** 的一个核心优势，即其内置的 **example types**，使 **agent** 能够为任务动态选择最合适的 **Extension**。
![[page_14_img_2.png]]![][image5]图 5\. 代理、扩展和 API 之间的一对多关系



请将此视为与软件开发人员在为用户问题设计解决方案（solutioning）时决定使用哪些 **API endpoints** 的方式相同。如果用户想预订机票，开发人员可能会使用 **Google Flights API**。如果用户想知道相对于其位置最近的咖啡店在哪里，开发人员可能会使用 **Google Maps API**。以同样的方式，**agent / model stack** 利用一组已知的 **Extensions** 来决定哪一个最契合用户的 **query**。 

如果您想了解 **Extensions** 的实际应用，可以在 **Gemini** 应用程序中通过 **Settings > Extensions** 并启用您想测试的任何项来进行尝试。例如，您可以启用 **Google Flights extension**，然后询问 **Gemini**：“帮我查看下周五从 **Austin** 到 **Zurich** 的航班。”

# **示例扩展**

为了简化 **Extensions** 的使用，**Google** 提供了一些开箱即用的 **extensions**，这些 **extensions** 可以被快速导入到您的项目中，并以极简的 **configurations** 即可使用。例如，**Snippet 1** 中的 **Code Interpreter extension** 允许您根据自然语言描述生成并运行 **Python** 代码。


```python
import vertexai
import pprint
PROJECT_ID = "YOUR_PROJECT_ID"
REGION = "us-central1"
vertexai.init(project=PROJECT_ID, location=REGION)
from vertexai.preview.extensions import Extension

extension_code_interpreter = Extension.from_hub("code_interpreter")
CODE_QUERY = """Write a python method to invert a binary tree in O(n) time."""
response = extension_code_interpreter.execute(
	operation_id = "generate_and_execute",
	operation_params = {"query": CODE_QUERY}
)
print("Generated Code:")
pprint.pprint({response['generated_code']})
# The above snippet will generate the following code.
Generated Code:
class TreeNode:
	def __init__(self, val=0, left=None, right=None(:
		self.val = val
		self.left = left
		self.right = right
		
	def invert_binary_tree(root):
	"""
	Inverts a binary tree.
	Args:
	root: The root of the binary tree.
	Returns:
	The root of the inverted binary tree.
	"""
	if not root:
	return None
	# Swap the left and right children recursively
	root.left, root.right =
	invert_binary_tree(root.right), invert_binary_tree(root.left)
	return root
	# Example usage:
	# Construct a sample binary tree
root = TreeNode(4)
root.left = TreeNode(2)
root.right = TreeNode(7)
root.left.left = TreeNode(1)
root.left.right = TreeNode(3)
root.right.left = TreeNode(6)
root.right.right = TreeNode(9)

# Invert the binary tree
inverted_root = invert_binary_tree(root)
```

  
总结而言，**Extensions** 为 **agents** 提供了一种以多种方式感知、交互并影响外部世界的方法。这些 **Extensions** 的选择和调用是由 **Examples** 的使用来引导的，而所有这些内容都被定义为 **Extension configuration** 的一部分。

# **函数**

在软件工程领域，**functions** 被定义为完成特定任务且可根据需要复用的独立代码模块。当软件开发人员编写程序时，他们通常会创建许多 **functions** 来执行各种任务。他们还会定义何时调用 **function_a** 还是 **function_b** 的逻辑，以及预期的输入和输出。

**Functions** 在 **agents** 领域的工作原理非常相似，只是我们可以将软件开发人员替换为 **model**。**model** 可以获取一组已知的 **functions**，并根据其 **specification** 决定何时使用各个 **Function**，以及该 **Function** 需要哪些 **arguments**。 **Functions** 与 **Extensions** 在几个方面存在差异，最显著的是： 
1. **model** 输出一个 **Function** 及其 **arguments**，但并不发起实时的 **API call**。 
2. **Functions** 在 **client-side** 执行，而 **Extensions** 则在 **agent-side** 执行。
  
再次以我们的 **Google Flights** 示例为例，一个简单的 **functions** 设置可能如图 7 中的示例所示。
![[page_19_img_1.png]]
![][image6]  
图 7\. 函数如何与外部 API 交互？

  
请注意，这里的主要区别在于：**Function** 和 **agent** 都不会直接与 **Google Flights API** 进行交互。那么，**API call** 实际上是如何发生的呢？ 

在使用 **functions** 时，调用实际 **API endpoint** 的逻辑和执行过程从 **agent** 侧卸载（offloaded），并交还给了 **client-side**（客户端）应用程序，如底部的图 8 和图 9 所示。这为开发人员提供了对应用程序数据流更细粒度的控制。开发人员选择使用 **functions** 而非 **Extensions** 的原因有很多，以下是一些常见的用例：

* **API calls** 需要在应用程序栈的其他层（而非直接在 **agent architecture** 流程中）发起（例如：中间件系统、前端框架等）。
* 由于安全或身份验证（**Authentication**）限制，阻止了 **agent** 直接调用 **API**（例如：**API** 未暴露在互联网上，或 **agent** 基础设施无法访问该 **API**）。 
* 存在时间或操作顺序上的限制，导致 **agent** 无法实时发起 **API calls**（即：批处理操作、人工干预审核等）。
* 需要对 **API Response** 应用 **agent** 无法执行的额外数据转换逻辑。例如，考虑到某个 **API endpoint** 未提供限制返回结果数量的过滤机制，在 **client-side** 使用 **Functions** 可以为开发人员提供执行这些转换的额外机会。 
* 开发人员希望在不为 **API endpoints** 部署额外基础设施的情况下迭代 **agent** 开发（即：**Function Calling** 可以起到 **API** “打桩（stubbing）”的作用）。

正如图 8 所示，虽然这两种方法在内部架构上的差异非常细微，但其提供的额外控制权以及与外部基础设施的解耦依赖，使得 **Function Calling** 成为对 **Developer** 而言极具吸引力的选择。

![[22365_19_Agents_v8_images/page_20_img_1.png]]![][image7]图 8\. 区分客户端和代理端对扩展和函数调用的控制



# **用例**

**model** 可以被用来调用 **functions**，以处理面向终端用户的复杂 **client-side** 执行流程。在这种场景下，**agent Developer** 可能不希望语言模型直接管理 **API** 的执行（如 **Extensions** 那样）。

让我们考虑这样一个示例：一个 **agent** 被训练为旅行管家（**travel concierge**），负责与想要预订度假行程的用户进行交互。其目标是让 **agent** 生成一个城市列表，以便我们可以在 **middleware application** 中使用该列表来为用户的行程规划下载图片、数据等。 

用户可能会说： “我想和家人去滑雪旅行，但我不确定该去哪里。” 

在向 **model** 发出的典型 **prompt** 中，输出结果可能如下所示： “当然，以下是一些您可以考虑进行家庭滑雪旅行的城市： 
• Crested Butte, Colorado, USA 
• Whistler, BC, Canada 
• Zermatt, Switzerland” 

虽然上述输出包含了我们需要的数据（城市名称），但其格式对于解析（parsing）而言并不理想。通过 **Function Calling**，我们可以教导 **model** 将此输出格式化为结构化样式（如 **JSON**），从而更方便其他系统进行解析。针对相同的用户输入 **prompt**，由 **Function** 生成的 **JSON** 输出示例可能如 **Snippet 5** 所示。

```json
function_call {
	name: "display_cities"
	args: {
		"cities": ["Crested Butte", "Whistler", "Zermatt"],
		"preferences": "skiing"
	}
}
```
代码片段 5：用于显示城市列表和用户偏好的示例函数调用有效负载

该 **JSON payload** 由 **model** 生成，随后被发送到我们的 **Client-side** 服务器，以便我们根据需求对其进行处理。在这个特定案例中，我们将调用 **Google Places API**，利用 **model** 提供的城市信息来查找 **Images**，然后将其作为格式化的 **rich content** 反馈给 **User**。请参考 **Figure 9** 中的 **sequence diagram**（时序图），它逐步详细展示了上述交互过程。

![[22365_19_Agents_v8_images/page_23_img_1.png]]
![][image8]  
图 9\. 函数调用生命周期的序列图

Figure 9 示例的结果是，利用 **model** 来“填空”，提供 **Client side UI** 调用 **Google Places API** 所需的参数。**Client side UI** 使用 **model** 在返回的 **Function** 中提供的参数来管理实际的 **API call**。这只是 **Function Calling** 的一个用例，还有许多其他场景值得考虑，例如： 
* 您希望 **language model** 建议一个可以在代码中使用的 **function**，但您不想在代码中包含凭据（**credentials**）。由于 **function calling** 并不运行该 **function**，因此您无需在包含 **function** 信息的代码中加入凭据。 
* 您正在运行可能耗时超过几秒钟的异步操作（**asynchronous operations**）。这些场景非常适合使用 **function calling**，因为它本身就是一种异步操作。 
* 您希望在与生成 **function calls** 及其 **arguments** 的系统不同的设备上运行 **functions**。 

关于 **functions** 需要记住的关键一点是，它们旨在为 **developer** 提供更大的控制权，不仅是对 **API calls** 执行的控制，还包括对整个应用程序数据流的全面控制。在 Figure 9 的示例中，**developer** 选择不将 **API** 信息返回给 **agent**，因为这与 **agent** 随后可能采取的行动无关。然而，基于应用程序的 **architecture**，将外部 **API call** 数据返回给 **agent** 以影响其未来的 **reasoning**、**logic** 和 **action** 选择可能是合理的。最终，应由应用程序 **developer** 根据具体应用场景来选择最合适的方法。

# **函数示例代码**

为了在上述滑雪度假场景中实现相应的输出，让我们构建每一个组件，使其能与我们的 **gemini-2.0-flash-001** 模型协同工作。 

首先，我们将 **display_cities** 函数定义为一个简单的 **Python** 方法。
```python
from typing import Optional
def display_cities(cities: list[str], preferences: Optional[str] = None):
	"""Provides a list of cities based on the user's search query and preferences.
	Args:
	preferences (str): The user's preferences for the search, like skiing,
	beach, restaurants, bbq, etc.
	cities (list[str]): The list of cities being recommended to the user.
	Returns:
	list[str]: The list of cities being recommended to the user.
	"""
	return cities
```
代码片段 7\. 构建工具，向模型发送用户查询并允许函数调用
  
接下来，我们将实例化我们的 **model**，构建 **Tool**，然后将用户的 **query** 和 **tools** 传递给 **model**。执行下方的代码将产生如 **code snippet** 底部所示的输出结果。
```python
from google.genai import Client, types
client = Client(
vertexai=True,
project="PROJECT_ID",
location="us-central1"
)
res = client.models.generate_content(
model="gemini-2.0-flash-001",
model="I'd like to take a ski trip with my family but I'm not sure where
to go?",
config=types.GenerateContentConfig(
tools=[display_cities],
automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
tool_config=types.ToolConfig(
function_calling_config=types.FunctionCallingConfig(mode='ANY')
)
)
)
print(f"Function Name: {res.candidates[0].content.parts[0].function_call.name}")
print(f"Function Args: {res.candidates[0].content.parts[0].function_call.args}")
> Function Name: display_cities
> Function Args: {'preferences': 'skiing', 'cities': ['Aspen', 'Park
City', 'Whistler']}
```
代码片段 7\. 构建工具，向模型发送用户查询并允许函数调用

总结而言，**functions** 提供了一个直观的框架，赋予了应用程序开发人员对 **data flow** 和 **system execution** 的细粒度控制能力，同时有效地利用 **agent/model** 进行关键输入生成。**Developers** 可以根据特定的 **application architecture** 需求，灵活选择是否通过返回外部数据让 **agent** 保持“**in the loop**”，或者将其省略。

# **数据存储**

想象一下，**language model** 就像一座庞大的藏书馆，包含着它的 **training data**。但与不断购入新书的图书馆不同，这座图书馆是静态的，仅保留其最初训练时所掌握的知识。这带来了一个挑战，因为现实世界的知识在不断演进。**Data Stores** 通过提供对更具动态性和时效性信息的访问，解决了这一局限，并确保 **model** 的响应保持 **grounded**（有据可查），具备真实性和相关性。 

考虑一个常见的场景：**developer** 可能需要向 **model** 提供少量额外数据，这些数据可能以电子表格或 **PDFs** 的形式存在。
![[22365_19_Agents_v8_images/page_27_img_1.png]]![][image10]图 10\. 智能体如何与结构化和非结构化数据交互？



**Data Stores** 允许开发人员以原始格式向 **agent** 提供额外数据，从而消除了耗时的数据转换、**model retraining** 或 **fine-tuning** 的需求。**Data Store** 将传入的文档转换为一组 **vector database embeddings**，**agent** 可以利用这些嵌入来提取所需信息，以补充其下一步行动或对用户的响应。

![[22365_19_Agents_v8_images/page_28_img_1.png]]![][image11]图 11\. 数据存储将代理连接到各种类型的新实时数据源。

# **实施与应用**
  
在 **Generative AI agents** 的语境下，**Data Stores** 通常被实现为开发人员希望 **agent** 在 **runtime** 期间能够访问的 **vector database**（向量数据库）。虽然我们在这里不会深入探讨 **vector database**，但需要理解的关键点是，它们以 **vector embeddings** 的形式存储数据，这是一种所提供数据的高维向量（high-dimensional vector）或数学表示。

近年来，将 **Data Store** 与语言模型结合使用的最著名案例之一是基于 **Retrieval Augmented Generation (RAG)** 的应用程序的实现。这些应用程序旨在通过赋予模型访问各种格式数据的能力，从而在基础训练数据之外扩展模型知识的广度和深度，这些格式包括：
* **Website content**（网页内容） 
* **Structured Data**（结构化数据），格式如 **PDF**、**Word Docs**、**CSV**、**Spreadsheets** 等 
* **Unstructured Data**（非结构化数据），格式如 **HTML**、**PDF**、**TXT** 等

![[page_29_img_1.png]]
![][image12]  
图 12\. 代理与数据存储之间的一对多关系，可以表示各种类型的预索引数据。

  
每个用户请求和 **agent** 响应循环的底层流程通常如 **Figure 13** 所示。 
1. **user query** 被发送到 **embedding model**，以为该查询生成 **embeddings**。 
2. 然后使用 **SCaNN** 等匹配算法，将 **query embeddings** 与 **vector database** 中的内容进行匹配。 
3. 匹配的内容以文本格式从 **vector database** 中检索出来并发送回 **agent**。 
4. **agent** 同时接收 **user query** 和检索到的内容，然后构建 **response** 或 **action**。
5. 最终响应（**final response**）被发送给用户。

![[page_30_img_1.png]]![][image13]图 13\. 基于 RAG 的应用程序中用户请求和代理响应的生命周期

最终结果是一个应用程序，它允许 **agent** 通过 **vector search** 将 **user’s query** 与已知的 **data store** 进行匹配，检索原始内容，并将其提供给 **orchestration layer** 和 **model** 进行进一步处理。下一步行动可能是向用户提供最终答案，或者是执行额外的 **vector search** 以进一步优化结果。 一个实现了结合 **ReAct reasoning/planning** 的 **RAG** 的 **agent** 交互示例请参见 **Figure 14**。
![[page_31_img_1.png]]
![][image14]  
图 14\. 基于 RAG 的 ReAct 推理/规划示例应用程序

# **工具概览**

总结而言，**extensions**、**functions** 和 **data stores** 构成了供 **agents** 在 **runtime** 使用的几种不同工具类型。每种工具都有其特定的用途，**agent developer** 可以根据需要决定将它们组合使用或独立使用。


|           | Extensions                                                                                                                                                                                                               | Function Calling                                                                                                                                                                                                                  | Data Stores                                                                                                                                                                                                                                                                                                               |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Execution | Agent-Side Execution                                                                                                                                                                                                     | Client-Side Execution                                                                                                                                                                                                             | Agent-Side Execution                                                                                                                                                                                                                                                                                                      |
| Use Case  | * 开发者希望 **agent** 控制与 **API endpoints** 的交互 <br>* 在利用原生预构建的 **Extensions**（如 **Vertex Search**、**Code Interpreter** 等）时非常有用 <br>* **Multi-hop planning**（多跳规划）和 **API** 调用（即：**agent** 的下一个动作取决于前一个动作或 **API call** 的输出） | <br>* 安全或 **Authentication**（身份验证）限制导致 **agent** 无法直接调用 **API**。 <br>* 时间约束或操作顺序（**order-of-operations**）约束导致 **agent** 无法实时发起 **API calls**（例如：批处理操作、**human-in-the-loop** 人工干预审核等）。 <br>* **API** 未暴露在互联网上，或 **Google** 系统无法访问。 | * **Developer** 希望利用以下任何数据类型实现 **Retrieval Augmented Generation (RAG)**： <br>* 来自预索引 **domains** 和 **URLs** 的 **Website Content** <br>* 格式如 **PDF**、**Word Docs**、**CSV**、**Spreadsheets** 等的 **Structured Data** <br>* **Relational / Non-Relational Databases** * 格式如 **HTML**、**PDF**、**TXT** 等的 **Unstructured Data** |


# **通过定向学习提升模型性能**

有效使用模型的关键一方面在于它们在生成输出时选择正确工具的能力，特别是在生产环境中大规模使用工具时。虽然通用训练有助于模型培养这种技能，但现实世界的场景通常需要超出训练数据的知识。请将此想象为基础烹饪技能与精通某一特定菜系之间的区别：两者都需要基础烹饪知识，但后者需要有针对性的学习，以获得更精细的结果。

为了帮助模型获取这类特定知识，存在几种方法：

* **In-context learning**：这种方法在 **inference** 时向通用模型提供 **prompt**、**tools** 以及 **few-shot examples**，使其能够“即时”（on the fly）学习如何以及何时针对特定任务使用这些 **tools**。**ReAct** 框架是这种方法在自然语言领域的一个例子。 
* **Retrieval-based in-context learning**：该技术通过从外部记忆中检索最相关的信息、**tools** 及相关示例，来动态填充模型的 **prompt**。这方面的例子包括 **Vertex AI extensions** 中的 “**Example Store**”，或者前面提到的基于 **data stores** 的 **RAG** 架构。 
* **Fine-tuning based learning**：该方法涉及在 **inference** 之前，使用更大的特定示例数据集对模型进行训练。这有助于模型在接收到任何用户查询之前，就理解何时以及如何应用某些 **tools**。

为了对每种有针对性的学习方法提供更深入的见解，让我们重新审视一下烹饪的比喻。

想象一位厨师从顾客那里收到了一份特定的食谱（**prompt**）、几种关键食材（相关 **tools**）以及一些示例菜肴（**few-shot examples**）。基于这些有限的信息以及厨师的通用烹饪知识，他们需要“即时”（**on the fly**）摸索出如何烹制出一道最符合该食谱和顾客偏好的菜肴。这就是 **in-context learning**。

现在，让我们想象这位厨师身处一个储备充足的储藏室（**external data stores**），里面装满了各种食材和食谱书（**examples** 和 **tools**）。厨师现在能够动态地从储藏室中挑选食材和食谱，从而更好地契合顾客的食谱和偏好。这使得厨师能够利用现有知识和新知识，制作出一道信息更丰富、更精细的菜肴。这就是 **retrieval-based in-context learning**。

最后，想象一下我们将厨师送回学校去学习一种或一系列新菜系（基于更大的特定示例数据集进行 **pre-training**）。这使得厨师在面对未来未曾见过的顾客食谱时，能够拥有更深刻的理解。如果我们希望厨师在特定菜系（**knowledge domains**）中表现卓越，这种方法是完美的。这就是 **fine-tuning based learning**。

每种方法在 **speed**、**cost** 和 **latency** 方面都具有独特的优缺点。然而，通过在 **agent framework** 中结合这些技术，我们可以充分利用各项优势并最大限度地减少其不足，从而实现一个更加 **robust** 且 **adaptable** 的解决方案。

# **LangChain 代理快速入门**

为了提供一个 **agent** 实际运作的现实世界可执行示例，我们将利用 **LangChain** 和 **LangGraph** 库构建一个快速原型。这些广受欢迎的开源库允许用户通过将一系列的 **logic**、**reasoning** 和 **tool calls** “链接（chaining）”在一起，来构建自定义 **agents** 以回答用户的查询。我们将使用我们的 **gemini-2.0-flash-001** 模型和一些简单的工具来回答来自用户的多阶段查询，如 **Snippet 8** 所示。 

我们使用的工具包括 **SerpAPI**（用于 **Google Search**）和 **Google Places API**。在执行 **Snippet 8** 中的程序后，您可以在 **Snippet 9** 中查看示例输出。

```python
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.tools import GooglePlacesTool
os.environ["SERPAPI_API_KEY"] = "XXXXX"
os.environ["GPLACES_API_KEY"] = "XXXXX"
@tool
def search(query: str):
"""Use the SerpAPI to run a Google Search."""
search = SerpAPIWrapper()
return search.run(query)
@tool
def places(query: str):
"""Use the Google Places API to run a Google Places Query."""
places = GooglePlacesTool()
return places.run(query)
model = ChatVertexAI(model="gemini-2.0-flash-001")
tools = [search, places]
query = "Who did the Texas Longhorns play in football last week? What is the
address of the other team's stadium?"
agent = create_react_agent(model, tools)
input = {"messages": [("human", query)]}
for s in agent.stream(input, stream_mode="values"):
message = s["messages"][-1]
if isinstance(message, tuple):
print(message)
else:
message.pretty_print()
```

示例 8\. 基于 LangChain 和 LangGraph 的代理及其工具

```python
============================== Human Message ================================
Who did the Texas Longhorns play in football last week? What is the address
of the other team's stadium?
================================= Ai Message =================================
Tool Calls: search
Args:
query: Texas Longhorns football schedule
================================ Tool Message ================================
Name: search
{...Results: "NCAA Division I Football, Georgia, Date..."}
================================= Ai Message =================================
The Texas Longhorns played the Georgia Bulldogs last week.
Tool Calls: places
Args:
query: Georgia Bulldogs stadium
================================ Tool Message ================================
Name: places
{...Sanford Stadium Address: 100 Sanford...}
================================= Ai Message =================================
The address of the Georgia Bulldogs stadium is 100 Sanford Dr, Athens, GA
30602, USA.
```

代码片段 9\. 代码片段 8 中程序的输出结果

虽然这是一个相当简单的 **agent** 示例，但它展示了 **Model**、**Orchestration** 以及 **tools** 这些基础组件如何协同工作以实现特定目标。在最后一部分中，我们将探讨这些组件如何在 **Google** 规模的托管产品中结合，例如 **Vertex AI agents** 和 **Generative Playbooks**。

# **使用 Vertex AI 代理的生产应用**

虽然本白皮书探讨了 **agents** 的核心组件，但构建 **production-grade**（生产级）应用程序需要将它们与 **user interfaces**（用户界面）、**evaluation frameworks**（评估框架）以及持续改进机制等额外工具相集成。**Google** 的 **Vertex AI** 平台通过提供一个包含前述所有基本要素的全托管环境，简化了这一过程。 利用 **natural language interface**（自然语言接口），开发人员可以快速定义其 **agents** 的关键要素——**goals**、**task instructions**、**tools**、用于 **task delegation**（任务授权）的 **sub-agents** 以及 **examples**——从而轻松构建所需的系统行为。此外，该平台还配备了一套开发工具，支持 **testing**、**evaluation**、衡量 **agent** 性能、**debugging** 以及提高所开发 **agents** 的整体质量。这使得开发人员能够专注于构建和优化其 **agents**，而 **infrastructure**、**deployment** 和 **maintenance** 的复杂性则由平台本身管理。 在 **Figure 15** 中，我们提供了一个在 **Vertex AI** 平台上构建的 **agent** 架构示例，该示例使用了 **Vertex Agent Builder**、**Vertex Extensions**、**Vertex Function Calling** 和 **Vertex Example Store** 等多种功能。该架构包含了构建 **production ready** 应用程序所需的许多核心组件。
![[22365_19_Agents_v8_images/page_39_img_1.png]]
![][image15]  
图 15\. 基于 Vertex AI 平台构建的端到端代理架构示例

您可以从我们的官方文档中尝试这种预构建代理架构的示例。
# **概括** 
在本白皮书中，我们讨论了 **Generative AI agents** 的基础构建模块、它们的组成，以及以 **cognitive architectures** 形式实现它们的有效方法。本白皮书的一些关键要点包括： 
1. **Agents** 通过利用 **tools** 访问实时信息、建议现实世界的行动，并自主地规划和执行复杂任务，从而扩展了 **language models** 的能力。**Agents** 可以利用一个或多个 **language models** 来决定何时以及如何进行状态迁移（transition through states），并使用外部 **tools** 来完成许多对于模型自身而言难以或不可能完成的复杂任务。 
2. **Agent** 运作的核心是 **orchestration layer**，这是一种构建 **reasoning**、**planning**、决策并引导其行动的 **cognitive architecture**。各种 **reasoning** 技术（如 **ReAct**、**Chain-of-Thought** 和 **Tree-of-Thoughts**）为 **orchestration layer** 提供了一个框架，使其能够接收信息、进行内部 **reasoning**，并生成明智的决策或响应。 
3. **Tools**（如 **Extensions**、**Functions** 和 **Data Stores**）是 **agents** 通往外部世界的钥匙，使它们能够与外部系统交互并访问超出其训练数据的知识。**Extensions** 在 **agents** 和外部 **APIs** 之间架起桥梁，实现了 **API calls** 的执行和实时信息的检索。**Functions** 通过分工为开发者提供了更精细的控制，允许 **agents** 生成可由 **client-side** 执行的 **Function** 参数。**Data Stores** 为 **agents** 提供了访问结构化或非结构化数据的能力，从而赋能数据驱动的应用程序。 

**Agents** 的未来充满了令人兴奋的进展，而我们目前仅仅触及了其潜力的冰山一角。随着 **tools** 变得更加精密以及 **reasoning** 能力的增强，**agents** 将有能力解决日益复杂的问题。此外，“**agent chaining**” 的战略性方法将继续保持势头。通过结合专门的 **agents**（每个 **agent** 在特定领域或任务中表现出色），我们可以创建一种 “**mixture of agent experts**” 的方法，能够在各个行业和问题领域交付卓越的结果。 

必须记住，构建复杂的 **agent architectures** 需要迭代的方法。实验和优化是为特定业务场景和组织需求寻找解决方案的关键。由于支撑其架构的 **foundational models** 具有生成特性，没有两个 **agent** 是完全相同的。然而，通过利用这些基础组件的优势，我们可以创建极具影响力的应用程序，扩展 **language models** 的能力并驱动现实世界的价值。

Endnotes:
1. Shafran, I., Cao, Y. et al., 2022, 'ReAct: Synergizing Reasoning and Acting in Language Models'. Available at:
https://arxiv.org/abs/2210.03629
2. Wei, J., Wang, X. et al., 2023, 'Chain-of-Thought Prompting Elicits Reasoning in Large Language Models'.
Available at: https://arxiv.org/pdf/2201.11903.pdf.
3. Wang, X. et al., 2022, 'Self-Consistency Improves Chain of Thought Reasoning in Language Models'.
Available at: https://arxiv.org/abs/2203.11171.
4. Diao, S. et al., 2023, 'Active Prompting with Chain-of-Thought for Large Language Models'. Available at:
https://arxiv.org/pdf/2302.12246.pdf.
5. Zhang, H. et al., 2023, 'Multimodal Chain-of-Thought Reasoning in Language Models'. Available at:
https://arxiv.org/abs/2302.00923.
6. Yao, S. et al., 2023, 'Tree of Thoughts: Deliberate Problem Solving with Large Language Models'. Available at:
https://arxiv.org/abs/2305.10601.
7. Long, X., 2023, 'Large Language Model Guided Tree-of-Thought'. Available at:
https://arxiv.org/abs/2305.08291.
8. Google. 'Google Gemini Application'. Available at: http://gemini.google.com.
9. Swagger. 'OpenAPI Specification'. Available at: https://swagger.io/specification/.
10. Xie, M., 2022, 'How does in-context learning work? A framework for understanding the differences from
traditional supervised learning'. Available at: https://ai.stanford.edu/blog/understanding-incontext/.
11. Google Research. 'ScaNN (Scalable Nearest Neighbors)'. Available at:
https://github.com/google-research/google-research/tree/master/scann.
12. LangChain. 'LangChain'. Available at: https://python.langchain.com/v0.2/docs/introduction/.