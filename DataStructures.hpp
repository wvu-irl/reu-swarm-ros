#pragma once
#include <stdio.h>
#include <stdlib.h>
/*
 *	Author: Henry Vos
 *
 *	The datatypes namespace is a namespace that looks to
 *	create efficient datastructures that are better than
 *	C++'s pitiful ones.
 */
namespace datatypes
{

/*
	 *	Linked List
	 *
	 *	The linked list (as the name suggests) is a variable length
	 *	data structure that is made of node structs.
	 *
	 *	3 nodes are tracked within the datastructure allowing for
	 *	iteration through the dataset quickly and not in O(n) time
	 *
	 */

template <class E>
class LinkedList
{
  private:
	/*
		 * Node structure
		 *
		 * used to contain all the data of the list
		 *
		 * @comp data = the actual data stored within the node
		 * @comp next = the pointer to the next node in the list
		 * @comp prev = the pointer to the previous node in the list
		 *
		 */
	struct node
	{
		E *data;
		void *next;
		void *prev;
	};

	struct node *head; // specifies the location of the start of the list
	struct node *mid;  // specofies some locaiton in the middle of the list
	struct node *tail; // specifies the location of the end of the list

	size_t size, midInd; // size = the number of elements within the list
						 // midInd = the index of the mid pointer

	/*
		 * Function used for traversing the datastructure efficiently
		 *
		 * this is used throughout the public functions to traverse to
		 * different nodes throughout the structure
		 *
		 * there is no error checking within this function
		 *
		 * @param i = the node to traverse
		 *
		 * @returns = the node found at the specified index
		 *
		 */
	struct node *findNode(size_t i)
	{
		// all these variables are squared becasue I don't want to import math.h
		size_t dHead = i * i;
		size_t dTail = (this->size - i) * (this->size - i);
		size_t dMid = (this->midInd - i) * (this->midInd - i);

		// setting up data for end cases
		if (dHead < dTail && dHead < dMid)
		{
			this->midInd = 0;
			this->mid = this->head;
		}
		else if (dTail < dHead && dTail < dMid)
		{
			this->midInd = this->size - 1;
			this->mid = this->tail;
		}
		// main traversal loop
		while (this->midInd != i && this->mid != NULL)
		{
			if (i < this->midInd)
			{
				this->midInd--;
				this->mid = (struct node *)this->mid->prev;
			}
			else // i > this->midInd
			{
				this->midInd++;
				this->mid = (struct node *)this->mid->next;
			}
		}
		return this->mid;
	}

  public:
	/*
		 * Constructor
		 *
		 * default only, sets everything to NULL or 0
		 *
		 */
	LinkedList()
	{
		this->head = NULL;
		this->mid = NULL;
		this->tail = NULL;
		this->size = 0;
		this->midInd = 0;
	}

	/**
	 * clears the list of all elements safely
	 * 
	 * empties list to size 0
	 */
	void clear()
	{
		while (this->size > 0)
		{
			this->removeFirst();
		}
	}

	/*
		 * Function returns weather the list is empty
		 *
		 * same as size == 0
		 */
	bool isEmpty()
	{
		return this->size <= 0;
	}

	/*
		 * adds a new element at the tail of the list
		 * tail index = size - 1
		 *
		 * @param val = pointer to the E value to add to the list
		 *
		 */
	void append(E *val)
	{
		// allocating space for the new node
		struct node *newNodePtr = (struct node *)malloc(sizeof(struct node));
		newNodePtr->next = NULL;
		newNodePtr->prev = NULL;
		newNodePtr->data = val;

		if (this->isEmpty())
		{
			// empty case
			this->mid = this->tail = this->head = newNodePtr;
		}
		else
		{
			// normal case
			newNodePtr->prev = this->tail;
			this->tail->next = newNodePtr;
			this->tail = newNodePtr;
		}
		this->size++;
	}

	/*
		 * Adds a new element to the head of the list
		 * head index = 0
		 *
		 * @param val = pointer to the E value to add to the list
		 *
		 */
	void prepend(E *val)
	{
		// allocating and initializing new data node
		struct node *newNodePtr = (struct node *)malloc(sizeof(struct node));
		newNodePtr->next = NULL;
		newNodePtr->prev = NULL;
		newNodePtr->data = val;

		if (this->isEmpty())
		{
			// empty list case
			this->head = this->tail = this->mid = newNodePtr;
		}
		else
		{
			// normal case
			this->midInd++;
			newNodePtr->next = this->head;
			this->head->prev = newNodePtr;
			this->head = newNodePtr;
		}
		this->size++;
	}

	/*
		 * Adds a new element to a specified location in the list
		 *
		 * value is bounded by the list bounds and will return NULL if exceeded
		 *
		 * @param i = the index the new value will have after insertion
		 * @param val = the pointer to the E value to add to the list
		 *
		 * @returns if the operation succeeded
		 *
		 */
	bool add(size_t i, E *val)
	{
		// error checking
		if (i > this->size || i < 0)
		{
			printf("List index out of bounds exception in add: %d\n", (int)i);
			return false;
		}
		// if the destination is in fact within the list
		if (this->isEmpty() || i == 0)
		{
			// empty or zero case
			this->prepend(val);
			return true;
		}
		else if (i == this->size)
		{
			// end case
			this->append(val);
			return true;
		}
		else
		{
			// normal case
			// allocating and setting data for new data node
			struct node *createdPtr = (struct node *)malloc(sizeof(struct node));
			createdPtr->data = val;

			// finding node that is currently holding this position
			struct node *foundPtr = findNode(i);
			// moving references
			((struct node *)foundPtr->prev)->next = createdPtr;
			createdPtr->prev = foundPtr->prev;
			createdPtr->next = foundPtr;
			foundPtr->prev = createdPtr;

			// updating mid index if neccessesery
			if (i < this->midInd)
				this->midInd++;

			this->size++;
			return true;
		}
	}

	/*
		 * Removes the first (head) element from the list
		 *
		 * @returns the removed value
		 *
		 */
	E *removeFirst()
	{
		if (this->size == 1)
		{
			E *ret = this->head->data;
			free(this->head);
			this->head = this->tail = NULL;
			this->size = 0;
			return ret;
		}

		// moving references
		((struct node *)this->head->next)->prev = NULL;
		struct node *tempPtr = this->head;
		this->head = (struct node *)this->head->next;
		E *retPtr = tempPtr->data; // saving data
		free(tempPtr);			   // freeing node
		// updating mid if necessesary
		if (this->midInd > 0)
			this->mid--;
		if (this->midInd == 0)
			this->mid = this->head;
		this->size--;
		return retPtr;
	}

	/*
		 * Removes the last (tail) element from the list
		 *
		 * @returns the removed value
		 */
	E *removeLast()
	{
		if (this->size == 1)
		{
			E *ret = this->head->data;
			free(this->head);
			this->head = this->tail = NULL;
			this->size = 0;
			return ret;
		}

		// moving references
		((struct node *)this->tail->prev)->next = NULL;
		struct node *tempPtr = this->tail;
		this->tail = (struct node *)this->tail->prev;
		E *retPtr = tempPtr->data; // saving data
		free(tempPtr);			   // freeing node
		// updating mid if necessesary
		if (this->midInd == this->size - 1)
		{
			this->mid = this->tail;
			this->midInd = this->size - 2;
		}
		this->size--;
		return retPtr;
	}

	/*
		 * Removes the element at the specified index
		 *
		 * bounded by list dimensions and will return null if exceeded
		 *
		 * @returns the removed value
		 *
		 */
	E *remove(size_t i)
	{
		// error checking
		if (i < 0 || i >= this->size)
		{
			printf("List index out of bounds exception in remove: %d\n", (int)i);
			return NULL;
		}

		if (this->size == 1)
		{
			E *ret = this->head->data;
			free(this->head);
			this->head = this->tail = NULL;
			this->size = 0;
			return ret;
		}

		// literal edge cases
		if (i == 0)
			return this->removeFirst();
		else if (i == this->size - 1)
			return this->removeLast();

		// finding target node
		struct node *foundPtr = findNode(i);
		E *retPtr = foundPtr->data; // saving data

		// moving references
		((struct node *)foundPtr->prev)->next = foundPtr->next;
		((struct node *)foundPtr->next)->prev = foundPtr->prev;
		// updating mid if needed
		if (this->midInd > i)
			this->midInd--;
		else if (this->midInd == i)
			this->mid = (struct node *)this->mid->next;

		free(foundPtr); // freeing node
		this->size--;

		return retPtr;
	}

	/*
		 * Gets the value of an element at an index
		 *
		 * index is bounded by list dimensions and returns NULL if exceeded
		 *
		 * @param i = the index to be viewed/returned
		 *
		 * @returns the value of the element at the specified index
		 */
	E *get(size_t i)
	{
		// error checking
		if (i < 0 || i > this->size - 1)
		{
			printf("List index out of bounds exception in get: %d\n", (int)i);
			return NULL;
		}
		// finding node
		struct node *foundPtr = findNode(i);
		return foundPtr->data;
	}

	/*
		 * Gives the dimensions of the structure
		 *
		 * @returns the nubmer of elements in the list
		 */
	size_t getSize()
	{
		return this->size;
	}

	bool containsValue(E *val)
	{
		for (size_t i = 0; i < this->size; i++)
		{
			if (*(findNode(i)->data) == *val)
			{
				return true;
			}
		}
		return false;
	}
};

} // namespace datatypes
