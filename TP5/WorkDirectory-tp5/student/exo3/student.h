#pragma once

#include <OPP.h>
#include <StudentWork.h>
#include <vector>
#include <iostream>

#include <previous/transform.h>
#include <previous/inclusive_scan.h>
#include <previous/exclusive_scan.h>
#include <previous/scatter.h>



class StudentWorkImpl: public StudentWork
{
public:

	bool isImplemented() const ;

	StudentWorkImpl() = default; 
	StudentWorkImpl(const StudentWorkImpl&) = default;
	~StudentWorkImpl() = default;
	StudentWorkImpl& operator=(const StudentWorkImpl&) = default;

	template<typename T>
	void run_radixSort_parallel(
		std::vector<T>& input,
		std::vector<T>& output
	) {
		// TODO
	}
	
	// Illustration de l'utilisation des itérateurs "counting" et "transform" définis dans OPP.h
	// Ils sont encore expérimental, mais bon ils font le boulot ;-)
	// TODO : acheter un antihistaminique
	void check() ;
};